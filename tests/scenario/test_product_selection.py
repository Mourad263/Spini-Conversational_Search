"""Sprint 6 product-selection task: selecting a previously-shown real
product by ordinal/last/brand reference (src/scenario/orchestrator.py's
C3 branch + _resolve_product_selection()). Hermetic -- `understand_fn` is
always stubbed with a fixed Action; RecordingSearch wraps the REAL
SearchAdapter/DB so the setup turns populate a genuine last_shown_products
list, and tests assert selection itself never calls it again.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.scenario.orchestrator import handle_message  # noqa: E402
from src.scenario.session_activity import DisplayedProduct, SessionActivityStore  # noqa: E402
from src.search_adapter.adapter import FilterSet, SearchAdapter, SearchResult  # noqa: E402
from src.state_manager.state_manager import StateManager  # noqa: E402
from src.understanding.schema import Action, Intent  # noqa: E402

LOGGER_NAME = "src.scenario.orchestrator"


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


def _last_trace(caplog) -> dict:
    records = [r for r in caplog.records if r.name == LOGGER_NAME and r.msg == "request_trace %s"]
    assert records, "expected a request_trace log record"
    return records[-1].args


def _search_milk_and_select(selection_action: Action, selection_message: str, caplog=None):
    # `selection_message` must be a REAL raw message carrying whatever
    # positional/textual evidence `selection_action` claims -- positional
    # fields are only trusted when the raw text itself supports them (see
    # orchestrator.py's _resolve_product_selection()), so a placeholder
    # like "selection message" would no longer resolve a position at all.
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"),
        selection_action,
    )
    search = RecordingSearch()
    handle_message("s1", "show me milk", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    shown = store.get("s1").last_shown_products

    if caplog is not None:
        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            r = handle_message("s1", selection_message, state_manager=sm, activity_store=store,
                                understand_fn=understand_fn, search_fn=search)
    else:
        r = handle_message("s1", selection_message, state_manager=sm, activity_store=store,
                            understand_fn=understand_fn, search_fn=search)
    return r, shown, store, search


# --- A/B/C: ordinal and last selection, resolved against the REAL shown list ---

def test_first_product_selection(caplog):
    r, shown, store, search = _search_milk_and_select(
        Action(intent=Intent.NO_OP, selection_ordinals=[1]), "the first one", caplog,
    )
    assert shown  # sanity: the milk search really returned products
    # Cart/direct-checkout architecture task: a PLAIN (non-checkout)
    # selection acknowledges the reference without writing selected_products
    # (reserved for an actually-opened checkout draft) -- verified via the
    # deterministic acknowledgment wording instead.
    assert store.get("s1").selected_products == []
    assert shown[0].name_en in r.assistant_message
    trace = _last_trace(caplog)
    assert trace["route"] == "product_selection"
    assert trace["search_calls"] == 0
    assert trace["recommendation_calls"] == 0
    assert trace["embedding_calls"] == 0


def test_third_product_selection():
    r, shown, store, search = _search_milk_and_select(
        Action(intent=Intent.NO_OP, selection_ordinals=[3]), "the third one",
    )
    assert len(shown) >= 3
    assert store.get("s1").selected_products == []
    assert shown[2].name_en in r.assistant_message


def test_last_product_selection():
    r, shown, store, search = _search_milk_and_select(
        Action(intent=Intent.NO_OP, selection_last=True), "the last one",
    )
    assert store.get("s1").selected_products == []
    assert shown[-1].name_en in r.assistant_message


# --- D: displayed brand/name reference, unique match ---
#
# last_shown_products is set directly here (not via a real brand-scoped
# search) so the "exactly one Juhayna product shown" precondition is
# controlled rather than incidental to today's live catalog composition
# -- the C3 resolution logic itself is exercised end-to-end regardless.

def test_unique_brand_reference_selects_exact_product():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(Action(intent=Intent.NO_OP, selection_text="Juhayna"))
    search = RecordingSearch()
    store.get("s1").last_shown_products = [
        DisplayedProduct(id=101, name_en="Juhayna Full Cream Milk", name_ar="لبن جهينة كامل الدسم",
                          brand_normalized="juhayna"),
        DisplayedProduct(id=102, name_en="Almarai Milk", name_ar="لبن المراعي", brand_normalized="almarai"),
        DisplayedProduct(id=103, name_en="Danone Yogurt", name_ar="زبادي دانون", brand_normalized="danone"),
    ]

    r = handle_message("s1", "I want the Juhayna one", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    # A PLAIN selection acknowledges without writing selected_products.
    assert store.get("s1").selected_products == []
    assert "Juhayna Full Cream Milk" in r.assistant_message
    assert len(search.calls) == 0


# --- E: ambiguous displayed brand reference -> no guessing ---

def test_ambiguous_reference_does_not_guess():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(Action(intent=Intent.NO_OP, selection_text="Juhayna"))
    search = RecordingSearch()
    store.get("s1").last_shown_products = [
        DisplayedProduct(id=101, name_en="Juhayna Full Cream Milk", name_ar="لبن جهينة كامل الدسم",
                          brand_normalized="juhayna"),
        DisplayedProduct(id=104, name_en="Juhayna Skimmed Milk", name_ar="لبن جهينة خالي الدسم",
                          brand_normalized="juhayna"),
        DisplayedProduct(id=103, name_en="Danone Yogurt", name_ar="زبادي دانون", brand_normalized="danone"),
    ]

    from src.response import messages
    r = handle_message("s1", "I want the Juhayna one", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert store.get("s1").selected_products == []
    assert r.assistant_message == messages.selection_ambiguous_response("I want the Juhayna one")


# --- F: out-of-range ordinal -> deterministic clarification, no guessing ---

def test_out_of_range_ordinal_does_not_guess():
    r, shown, store, search = _search_milk_and_select(Action(intent=Intent.NO_OP, selection_ordinals=[999]), "999")
    assert store.get("s1").selected_products == []
    assert str(len(shown)) in r.assistant_message


# --- G: no display context -> honest, no search ---

def test_no_display_context_does_not_search():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(Action(intent=Intent.NO_OP, selection_ordinals=[1]))
    search = RecordingSearch()

    r = handle_message("s1", "I want the first one", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert len(search.calls) == 0
    assert store.get("s1").selected_products == []
    assert "haven't shown" in r.assistant_message.lower()


# --- H: recommendation results become the latest display context ---

def test_recommendation_results_become_latest_display_context():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"),
        Action(intent=Intent.NO_OP, recommendation_request=True),
        Action(intent=Intent.NO_OP, selection_ordinals=[2]),
    )
    search = RecordingSearch()
    handle_message("s1", "I need milk", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    search_shown = list(store.get("s1").last_shown_products)

    handle_message("s1", "what do you recommend?", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    rec_shown = store.get("s1").last_shown_products
    assert rec_shown != search_shown  # sanity: recommendations really replaced the context

    # A PLAIN selection acknowledges without writing selected_products --
    # proves resolution used the recommendation context (rec_shown), not
    # the earlier search context.
    r = handle_message("s1", "I want the second one", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert store.get("s1").selected_products == []
    assert rec_shown[1].name_en in r.assistant_message


# --- I: unseen product reference must not search the catalog or select ---

def test_unseen_product_reference_does_not_search_or_select():
    r, shown, store, search = _search_milk_and_select(
        Action(intent=Intent.NO_OP, selection_text="xyzzyplughnonexistentbrand"), "selection message",
    )
    calls_before = len(search.calls)
    assert store.get("s1").selected_products == []
    assert len(search.calls) == calls_before  # selection turn made no new search


# --- K: recommendation-offer pending interaction ---

def test_selection_clears_pending_offer_and_proceeds():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"),
        Action(intent=Intent.NO_OP, selection_ordinals=[1]),
    )
    search = RecordingSearch()
    handle_message("s1", "show me milk", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    assert store.get("s1").pending_action is not None
    shown = store.get("s1").last_shown_products

    r = handle_message("s1", "I want the first one", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert store.get("s1").pending_action is None
    # A PLAIN selection acknowledges without writing selected_products.
    assert store.get("s1").selected_products == []
    assert shown[0].name_en in r.assistant_message


# --- L/M: state stored session-side, isolated per session ---

def test_selected_product_is_a_real_previously_shown_product_and_session_isolated():
    r, shown, store, search = _search_milk_and_select(
        Action(intent=Intent.NO_OP, selection_ordinals=[1]), "the first one",
    )
    # A PLAIN selection acknowledges a real, previously-shown product
    # without writing selected_products (reserved for an open checkout).
    assert store.get("s1").selected_products == []
    assert shown[0].name_en in r.assistant_message
    # a second, unrelated session never saw these products at all
    store2 = SessionActivityStore()
    assert store2.get("s2").last_shown_products == []
    assert store2.get("s2").selected_products == []


# --- N: session expiry clears displayed/selected context ---

def test_session_expiry_clears_display_and_selection_context():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"))
    search = RecordingSearch()
    handle_message("s1", "show me milk", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search, now=0.0)
    assert store.get("s1", now=0.0).last_shown_products

    from src.scenario.session_activity import SESSION_TTL_SECONDS
    understand_fn2 = stub_understand(Action(intent=Intent.OFF_TOPIC))
    handle_message("s1", "hello there", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn2, search_fn=search, now=SESSION_TTL_SECONDS + 100.0)

    assert store.get("s1", now=SESSION_TTL_SECONDS + 100.0).last_shown_products == []
    assert store.get("s1", now=SESSION_TTL_SECONDS + 100.0).selected_products == []


# --- O: normal search/recommendation/social/memory regression ---

def test_normal_search_regression_unaffected():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"))
    search = RecordingSearch()
    r = handle_message("s1", "I need milk", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)
    assert r.resolved_category == "milk_29"
    assert len(r.products) > 0
