"""Named (selection_text) product selection fix: the resolver previously
matched only by exact substring containment (_text_matches_displayed_product),
which had zero tolerance for the ordinary spelling variation a real
conversation produces ("fa yoghart" for "Fa Yoghurt..."). Replaced by
fuzzy scoring (_selection_text_score, rapidfuzz.fuzz.WRatio) against ONLY
last_shown_products -- never the wider catalog. Hermetic -- `understand_fn`
is always stubbed; controlled DisplayedProduct fixtures are set directly
(same pattern as tests/scenario/test_product_selection.py's brand-reference
tests) so exact product names are deterministic, not incidental to today's
live catalog composition.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.scenario.orchestrator import handle_message  # noqa: E402
from src.scenario.session_activity import DisplayedProduct, SessionActivityStore  # noqa: E402
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


_FA = DisplayedProduct(id=201, name_en="Fa Yoghurt Shower Cream With Vanilla Honey",
                        name_ar="فا كريم استحمام بالزبادي والعسل والفانيليا", brand_normalized="fa",
                        price=45.0, special_price=45.0)  # a real price -- checkout's
# unknown-price guard (src/pricing.py's effective_price()) would otherwise block any
# checkout scenario using this fixture; earlier revisions of this fixture predated
# that guard, when this field was still unused by any test in this file
_DOVE_STICK = DisplayedProduct(id=202, name_en="Dove Deodorant Stick", name_ar="دوف مزيل عرق استيك",
                                brand_normalized="dove")
_DOVE_SPRAY = DisplayedProduct(id=203, name_en="Dove Deodorant Spray", name_ar="دوف مزيل عرق سبراي",
                                brand_normalized="dove")


def _primed_session(session_id: str = "s1") -> tuple[StateManager, SessionActivityStore]:
    sm, store = StateManager(), SessionActivityStore()
    store.get(session_id).last_shown_products = [_FA, _DOVE_STICK, _DOVE_SPRAY]
    return sm, store


# --- A: partial named selection (brief description, not the full title) ---

def test_partial_named_selection_matches_full_title():
    sm, store = _primed_session()
    understand_fn = stub_understand(Action(intent=Intent.NO_OP, selection_text="fa yoghurt"))
    search = RecordingSearch()

    r = handle_message("s1", "i will buy the fa yoghurt", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    # Cart/direct-checkout architecture task: a PLAIN (non-checkout)
    # selection acknowledges the reference without writing selected_products
    # (reserved for an actually-opened checkout draft).
    assert store.get("s1").selected_products == []
    assert _FA.name_en in r.assistant_message
    assert len(search.calls) == 0


# --- B: minor typo in named selection still resolves ---

def test_minor_typo_in_named_selection_still_resolves():
    sm, store = _primed_session()
    understand_fn = stub_understand(Action(intent=Intent.NO_OP, selection_text="fa yoghart"))
    search = RecordingSearch()

    r = handle_message("s1", "i will buy the fa yoghart", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert store.get("s1").selected_products == []
    assert _FA.name_en in r.assistant_message
    assert len(search.calls) == 0


# --- C: named selection after a zero-result turn in between ---

def test_named_selection_after_zero_result_turn():
    sm, store = _primed_session()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk", price_max=0.01),
        Action(intent=Intent.NO_OP, selection_text="fa yoghart"),
    )
    search = RecordingSearch()

    r1 = handle_message("s1", "show me milk under 0.01", state_manager=sm, activity_store=store,
                         understand_fn=understand_fn, search_fn=search)
    assert r1.zero_result is True  # sanity: this really was a zero-result turn
    assert list(store.get("s1").last_shown_products) == [_FA, _DOVE_STICK, _DOVE_SPRAY]  # unchanged

    r = handle_message("s1", "i will buy the fa yoghart", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)
    assert store.get("s1").selected_products == []
    assert _FA.name_en in r.assistant_message


# --- D: named selection + checkout in the SAME turn ---

def test_named_selection_and_checkout_same_turn():
    sm, store = _primed_session()
    understand_fn = stub_understand(
        Action(intent=Intent.NO_OP, selection_text="fa yoghart", checkout_request=True),
    )
    search = RecordingSearch()

    r = handle_message("s1", "i will buy the fa yoghart", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert store.get("s1").selected_products[0].product.id == _FA.id
    assert store.get("s1").checkout_active is True
    assert r.checkout_form is not None
    assert r.checkout_form.lines[0].id == _FA.id
    assert len(search.calls) == 0


# --- E: ambiguity safety -- one shared token across two shown products
# must never be arbitrarily resolved ---

def test_ambiguous_shared_brand_token_does_not_guess():
    sm, store = _primed_session()
    understand_fn = stub_understand(Action(intent=Intent.NO_OP, selection_text="dove"))
    search = RecordingSearch()
    from src.response import messages

    r = handle_message("s1", "the dove one", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert store.get("s1").selected_products == []
    assert r.assistant_message == messages.selection_ambiguous_response("the dove one")


# --- F: text naming no shown product -> honest failure, never a catalog
# search fallback ---

def test_unmatched_named_selection_never_falls_back_to_catalog_search():
    sm, store = _primed_session()
    understand_fn = stub_understand(Action(intent=Intent.NO_OP, selection_text="xyzzyplughnonexistentbrand"))
    search = RecordingSearch()

    r = handle_message("s1", "i'll take the xyzzyplugh one", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert store.get("s1").selected_products == []
    assert len(search.calls) == 0


# --- G: a new successful display supersedes the old set -- a named
# reference to the OLD set no longer resolves ---

def test_new_successful_display_supersedes_named_reference_to_old_set():
    sm, store = _primed_session()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"),
        Action(intent=Intent.NO_OP, selection_text="fa yoghurt"),
    )
    search = RecordingSearch()

    handle_message("s1", "show me milk", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    assert _FA not in store.get("s1").last_shown_products  # sanity: real search replaced it

    from src.response import messages
    r = handle_message("s1", "i will buy the fa yoghurt", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert store.get("s1").selected_products == []
    assert r.assistant_message == messages.selection_not_found_response("i will buy the fa yoghurt")


# --- H: ordinal selection regression -- unaffected by the fuzzy-text change ---

def test_ordinal_selection_regression():
    sm, store = _primed_session()
    understand_fn = stub_understand(Action(intent=Intent.NO_OP, selection_ordinals=[2]))
    search = RecordingSearch()

    r = handle_message("s1", "the second one", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert store.get("s1").selected_products == []
    assert _DOVE_STICK.name_en in r.assistant_message
