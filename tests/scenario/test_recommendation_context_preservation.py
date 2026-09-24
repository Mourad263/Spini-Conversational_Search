"""Sprint 6 search+recommendation context task: a real live-Groq call
confirmed "recommend me more" (after an active "chips, price>=20" state)
comes back as intent=SEARCH with every field null except
recommendation_request=True -- not NO_OP. Unrepaired, this state-changing
SEARCH went through StateManager.apply() and REPLACED the active
FilterSet with an empty one, wiping "chips"/price_min=20 into an
unconstrained whole-catalog browse, which then also leaked into
recommendation candidates (both surfaced totally unrelated products).

Two fixes, both general (operate on Action/state shape, never on
message text or product-specific values):

1. src/scenario/decision.py's normalize_recommendation_intent() (the
   single Action-normalization function for recommendation_request=True,
   covering this direction and its mirror-image sibling -- a NO_OP that
   should have been a real intent because a price WAS stated): a state-
   changing intent (search/add_filter/modify_filter) that states
   literally no new concrete value AND asks for recommendations is
   repaired to NO_OP, so StateManager.apply() is never reached and the
   active state survives untouched.
2. src/scenario/orchestrator.py's NO_OP/recommendation branch: when the
   recommendation-exploration policy applied no new price band
   (nothing new to explore), the redundant "products" re-search/re-
   display is skipped entirely -- only recommendations answer a pure
   continuation turn.

Hermetic -- `understand_fn` is stubbed with the REAL confirmed Action
shape for the final turn; RecordingSearch wraps the real SearchAdapter/DB.
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


def _chips_above_20_then_recommend(final_action: Action):
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="chips", category_hint="chips"),
        Action(intent=Intent.ADD_FILTER, price_min=20.0),
        final_action,
    )
    search = RecordingSearch()
    handle_message("s1", "i want bag of chips", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    handle_message("s1", "above 20", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    state_before = sm.get_state("s1")
    r = handle_message("s1", "recommend me more", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)
    return r, sm.get_state("s1"), state_before, search


# --- A: the REAL confirmed Action shape (intent=SEARCH, all fields null,
# recommendation_request=True) is repaired to a NO_OP that preserves state ---

def test_misclassified_search_recommendation_preserves_active_state():
    real_action = Action(intent=Intent.SEARCH, recommendation_request=True)
    r, state_after, state_before, search = _chips_above_20_then_recommend(real_action)

    assert state_after == state_before  # NOT wiped -- still query_text="chips", price_min=20
    assert state_after.query_text == "chips"
    assert state_after.price_min == 20.0
    assert r.intent == "no_op"


# --- B: no redundant/generic "products" list for a pure continuation turn ---

def test_pure_recommendation_continuation_shows_no_generic_product_list():
    real_action = Action(intent=Intent.SEARCH, recommendation_request=True)
    r, *_ = _chips_above_20_then_recommend(real_action)

    assert r.products == []
    assert r.assistant_message == "Here are some other options you may like."


# --- B2: recommendation-continuation search-trace fix -- no ordinary search
# is repeated for a pure continuation; current_state hasn't changed since
# "above 20" already searched it, so last_shown_products is reused instead ---

def test_pure_recommendation_continuation_makes_no_redundant_search():
    real_action = Action(intent=Intent.SEARCH, recommendation_request=True)
    r, _, _, search = _chips_above_20_then_recommend(real_action)

    assert r.recommendations  # still produced real recommendations
    assert len(search.calls) == 2  # only "chips" + "above 20", no 3rd call for "recommend me more"


# --- C: recommendations reflect the CURRENT (preserved) topic + price, not
# an unconstrained catalog browse ---

def test_recommendations_reflect_preserved_topic_and_price():
    real_action = Action(intent=Intent.SEARCH, recommendation_request=True)
    r, *_ = _chips_above_20_then_recommend(real_action)

    assert r.recommendations  # non-empty
    for p in r.recommendations:
        assert "chip" in p.name_en.lower()
        assert p.price is None or p.price >= 20.0


# --- D: a genuinely new facet in the SAME shape is NOT swallowed -- the
# repair only fires when literally nothing new is stated ---

def test_search_with_a_real_new_facet_is_not_converted_to_no_op():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="chips", category_hint="chips"),
        Action(intent=Intent.ADD_FILTER, price_min=20.0),
        Action(intent=Intent.SEARCH, raw_query_text="chocolate", recommendation_request=True),
    )
    search = RecordingSearch()
    handle_message("s1", "i want bag of chips", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    handle_message("s1", "above 20", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)

    r = handle_message("s1", "actually show me chocolate", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert sm.get_state("s1").query_text == "chocolate"  # a real new search DID take effect
    assert r.intent == "search"
