"""Sprint 6 follow-up: a failed (zero-result or out-of-catalog) search must
never erase the last successfully displayed, still-selectable product set
(src/scenario/session_activity.py's last_shown_products). Investigation
confirmed this is already correctly guarded at every
`SessionActivityStore.set_last_shown_products()` call site in
orchestrator.py (only called when a fresh result is genuinely non-empty)
and that the out-of-catalog branch returns early without touching it at
all -- these tests lock that existing contract in with explicit
regression coverage, since none existed before. Hermetic -- `understand_fn`
is always stubbed; RecordingSearch wraps the REAL SearchAdapter/DB.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.scenario.orchestrator import handle_message  # noqa: E402
from src.scenario.session_activity import SessionActivityStore  # noqa: E402
from src.search_adapter.adapter import FilterSet, SearchAdapter, SearchResult  # noqa: E402
from src.state_manager.state_manager import StateManager  # noqa: E402
from src.understanding.schema import Action, Intent  # noqa: E402


def stub_understand(*actions: Action):
    it = iter(actions)

    def _stub(message: str, history: list[str] | None = None) -> Action:
        return next(it)

    return _stub


class RecordingSearch:
    def __init__(self) -> None:
        self.calls: list[FilterSet] = []

    def __call__(self, filters: FilterSet, limit: int, offset: int) -> SearchResult:
        self.calls.append(filters)
        return SearchAdapter().search(filters, limit=limit, offset=offset)


# --- A: a genuinely zero-result (but not out-of-catalog) search preserves
# the previous successful display, and selection against it still works ---

def test_zero_result_search_preserves_last_shown_and_selection_still_works():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"),
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk", price_max=0.01),
        Action(intent=Intent.NO_OP, selection_ordinals=[2]),
    )
    search = RecordingSearch()
    handle_message("s1", "show me milk", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    shown = list(store.get("s1").last_shown_products)
    assert shown  # sanity: something was really displayed

    r2 = handle_message("s1", "show me milk under 0.01", state_manager=sm, activity_store=store,
                         understand_fn=understand_fn, search_fn=search)
    assert r2.zero_result is True
    assert r2.likely_out_of_catalog is False
    assert list(store.get("s1").last_shown_products) == shown  # unchanged

    # Cart/direct-checkout architecture task: a PLAIN (non-checkout)
    # selection acknowledges the shown-product reference without writing
    # selected_products (that is reserved for an actually-opened checkout
    # draft) -- verified via the deterministic acknowledgment wording
    # instead, which still proves resolution used the PRESERVED last_shown
    # list (shown[1]), not a stale/empty one.
    r3 = handle_message("s1", "the second one", state_manager=sm, activity_store=store,
                         understand_fn=understand_fn, search_fn=search)
    assert r3.intent == "product_selection"
    assert shown[1].name_en in r3.assistant_message
    assert store.get("s1").selected_products == []


# --- B: an out-of-catalog turn preserves the previous successful display,
# and selection against it still works ---

def test_out_of_catalog_preserves_last_shown_and_selection_still_works():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"),
        Action(intent=Intent.SEARCH, raw_query_text="laptop charger"),
        Action(intent=Intent.NO_OP, selection_ordinals=[1]),
    )
    search = RecordingSearch()
    handle_message("s1", "show me milk", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    shown = list(store.get("s1").last_shown_products)

    r2 = handle_message("s1", "i need a laptop charger", state_manager=sm, activity_store=store,
                         understand_fn=understand_fn, search_fn=search)
    assert r2.likely_out_of_catalog is True
    assert list(store.get("s1").last_shown_products) == shown  # unchanged
    calls_before = len(search.calls)  # out-of-catalog must not have searched at all

    # A PLAIN selection acknowledges without writing selected_products
    # (see the sibling zero-result test above for the full rationale).
    r3 = handle_message("s1", "the first one", state_manager=sm, activity_store=store,
                         understand_fn=understand_fn, search_fn=search)
    assert r3.intent == "product_selection"
    assert shown[0].name_en in r3.assistant_message
    assert len(search.calls) == calls_before  # selection made no new search either


# --- C: a new SUCCESSFUL search (after an intervening zero-result turn)
# still correctly supersedes the earlier display ---

def test_new_successful_search_after_zero_result_still_supersedes():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"),
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk", price_max=0.01),
        Action(intent=Intent.SEARCH, raw_query_text="chocolate", category_hint="chocolate"),
        Action(intent=Intent.NO_OP, selection_ordinals=[1]),
    )
    search = RecordingSearch()
    handle_message("s1", "show me milk", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    handle_message("s1", "show me milk under 0.01", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    handle_message("s1", "show me chocolate", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    choc_shown = list(store.get("s1").last_shown_products)

    # A PLAIN selection acknowledges without writing selected_products
    # (see the first test in this file for the full rationale).
    r = handle_message("s1", "the first one", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)
    assert r.intent == "product_selection"
    assert choc_shown[0].name_en in r.assistant_message


# --- D: RESET supersedes the previous display with a fresh one -- the old
# set is no longer what a later selection resolves against ---

def test_reset_supersedes_previous_display():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"),
        Action(intent=Intent.RESET),
    )
    search = RecordingSearch()
    handle_message("s1", "show me milk", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    shown_before = [p.id for p in store.get("s1").last_shown_products]

    handle_message("s1", "start over", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    shown_after = [p.id for p in store.get("s1").last_shown_products]

    assert shown_after != shown_before
    assert store.get("s1").selected_products == []


# --- E: same-turn "select X and checkout" remains correct even with a
# zero-result turn between the display and the selection+checkout turn ---

def test_same_turn_selection_and_checkout_after_intervening_zero_result():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"),
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk", price_max=0.01),
        Action(intent=Intent.NO_OP, selection_ordinals=[2], checkout_request=True),
    )
    search = RecordingSearch()
    handle_message("s1", "show me milk", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    shown = list(store.get("s1").last_shown_products)
    handle_message("s1", "show me milk under 0.01", state_manager=sm, activity_store=store,
                   understand_fn=understand_fn, search_fn=search)

    r = handle_message("s1", "I'll take the second one and checkout", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert store.get("s1").selected_products[0].product.id == shown[1].id
    assert r.checkout_form is not None
    assert r.checkout_form.lines[0].id == shown[1].id
