"""Sprint 6 recommendation-offer + pending-action task. Hermetic --
`understand_fn` here RAISES if called at all for acceptance/rejection
turns, proving zero LLM calls (stronger than just checking
trace.llm_calls==0). RecordingSearch wraps the REAL SearchAdapter/DB;
RecommendationService is the real one too (real DB/clustering), same
pattern as tests/scenario/test_out_of_catalog_grounding.py and
test_recommendation_integration.py.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.scenario.orchestrator import handle_message  # noqa: E402
from src.scenario.session_activity import SessionActivityStore  # noqa: E402
from src.search_adapter.adapter import FilterSet, SearchAdapter, SearchResult  # noqa: E402
from src.state_manager.state_manager import StateManager  # noqa: E402
from src.understanding.schema import Action, Intent  # noqa: E402

LOGGER_NAME = "src.scenario.orchestrator"


def understand_must_not_be_called(message: str, history: list[str] | None = None) -> Action:
    raise AssertionError(f"understand() was called: {message!r}")


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


# --- A: successful normal search creates the offer, no RecommendationService call ---

def test_successful_search_creates_offer_without_calling_recommendation_service(caplog):
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"))
    search = RecordingSearch()
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        r = handle_message("s1", "show me milk", state_manager=sm, activity_store=store,
                            understand_fn=understand_fn, search_fn=search)

    assert len(r.products) > 0
    assert "recommend" in r.assistant_message.lower() or "أرشح" in r.assistant_message
    assert store.get("s1").pending_action is not None
    assert store.get("s1").pending_action.kind == "recommendation_offer"
    trace = _last_trace(caplog)
    assert trace["recommendation_calls"] == 0


# --- B: English acceptance ---

def test_english_acceptance_triggers_recommendation_service_once(caplog):
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"))
    search = RecordingSearch()
    handle_message("s1", "show me milk", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        r = handle_message("s1", "yes", state_manager=sm, activity_store=store,
                            understand_fn=understand_must_not_be_called, search_fn=search)

    assert isinstance(r.recommendations, list) and len(r.recommendations) > 0
    assert store.get("s1").pending_action is None
    trace = _last_trace(caplog)
    assert trace["route"] == "recommendation_accept"
    assert trace["llm_calls"] == 0
    assert trace["search_calls"] == 0
    assert trace["recommendation_calls"] == 1


# --- C: Arabic acceptance ---

def test_arabic_acceptance_same_behavior():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"))
    search = RecordingSearch()
    handle_message("s1", "show me milk", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)

    r = handle_message("s1", "ايوه", state_manager=sm, activity_store=store,
                        understand_fn=understand_must_not_be_called, search_fn=search)

    assert len(r.recommendations) > 0
    assert store.get("s1").pending_action is None


# --- D: rejection ---

def test_rejection_clears_pending_no_expensive_calls():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"))
    search = RecordingSearch()
    handle_message("s1", "show me milk", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    calls_before = len(search.calls)

    r = handle_message("s1", "no", state_manager=sm, activity_store=store,
                        understand_fn=understand_must_not_be_called, search_fn=search)

    assert r.assistant_message == "No problem. What else can I help you find?"
    assert r.recommendations == []
    assert store.get("s1").pending_action is None
    assert len(search.calls) == calls_before


# --- E: Arabic rejection, no expensive calls ---

def test_arabic_rejection(caplog):
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"))
    search = RecordingSearch()
    handle_message("s1", "show me milk", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        r = handle_message("s1", "لا", state_manager=sm, activity_store=store,
                            understand_fn=understand_must_not_be_called, search_fn=search)

    assert r.assistant_message == "تمام، تحب أدور لك على حاجة تانية؟"
    trace = _last_trace(caplog)
    assert trace["llm_calls"] == 0
    assert trace["search_calls"] == 0
    assert trace["recommendation_calls"] == 0


# --- F: mixed acceptance + shopping is not swallowed ---

def test_mixed_yes_and_shopping_not_swallowed():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"),
        Action(intent=Intent.SEARCH, raw_query_text="cheese", category_hint="cheese"),
    )
    search = RecordingSearch()
    handle_message("s1", "show me milk", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    milk_offer = store.get("s1").pending_action

    r = handle_message("s1", "yes, show me cheese", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert r.resolved_category == "cheese"
    # the stale milk offer was consumed (not left dangling) -- the cheese
    # search that actually ran created its OWN fresh offer in its place
    # (Sprint 6 section 7: "replace it when a new successful search
    # generates a new offer"), never the old milk one.
    cheese_offer = store.get("s1").pending_action
    assert cheese_offer is not None
    assert cheese_offer.shown_product_ids != milk_offer.shown_product_ids


# --- G: stale pending offer does not hijack a later unrelated turn ---

def test_stale_offer_cleared_by_next_turn_regardless_of_content():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"),
        Action(intent=Intent.OFF_TOPIC),
        Action(intent=Intent.OFF_TOPIC),  # the LATER bare "yes" -- reaches the real LLM now
    )
    search = RecordingSearch()
    handle_message("s1", "show me milk", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    assert store.get("s1").pending_action is not None

    handle_message("s1", "what's the weather?", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    assert store.get("s1").pending_action is None

    # a LATER bare "yes" must not be treated as accepting the old offer --
    # no pending action exists anymore, so it correctly falls through to
    # the real understand() workflow (stubbed here) instead of RecommendationService.
    r = handle_message("s1", "yes", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)
    assert r.recommendations == []


# --- H: zero-result search creates no offer ---

def test_zero_result_search_creates_no_offer():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="xyzzyplughnonsenseword", category_hint="xyzzyplughnonsenseword"),
    )
    search = RecordingSearch()
    r = handle_message("s1", "xyzzyplughnonsenseword", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert r.zero_result is True
    assert store.get("s1").pending_action is None


# --- I: out-of-catalog creates no offer ---

def test_out_of_catalog_creates_no_offer():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(Action(intent=Intent.SEARCH, raw_query_text="كوتش عربية"))
    search = RecordingSearch()
    r = handle_message("s1", "انا عاوز كوتش عربيه", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert r.likely_out_of_catalog is True
    assert store.get("s1").pending_action is None


# --- J: memory recall creates no offer ---

def test_memory_recall_creates_no_offer():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"),
        Action(intent=Intent.NO_OP, memory_recall="first"),
    )
    search = RecordingSearch()
    handle_message("s1", "I need milk", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    store.get("s1").pending_action = None  # the search itself created one; clear to isolate this check

    handle_message("s1", "what did I first ask for?", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)

    assert store.get("s1").pending_action is None


# --- K: social response creates no offer ---

def test_social_response_creates_no_offer():
    sm, store = StateManager(), SessionActivityStore()
    search = RecordingSearch()
    handle_message("s1", "hello", state_manager=sm, activity_store=store,
                   understand_fn=understand_must_not_be_called, search_fn=search)

    assert store.get("s1").pending_action is None


# --- L: direct explicit recommendation request still works normally ---

def test_direct_recommendation_request_regression():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk", recommendation_request=True),
    )
    search = RecordingSearch()
    r = handle_message("s1", "show me milk and recommend something similar", state_manager=sm,
                        activity_store=store, understand_fn=understand_fn, search_fn=search)

    assert isinstance(r.recommendations, list)
    # an explicit recommendation_request turn must NOT also queue a NEW offer
    assert store.get("s1").pending_action is None


# --- O: session expiry clears pending_action ---

def test_session_expiry_clears_pending_action():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"))
    search = RecordingSearch()
    handle_message("s1", "show me milk", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search, now=0.0)
    assert store.get("s1", now=0.0).pending_action is not None

    from src.scenario.session_activity import SESSION_TTL_SECONDS
    handle_message("s1", "hello", state_manager=sm, activity_store=store,
                    understand_fn=understand_must_not_be_called, search_fn=search,
                    now=SESSION_TTL_SECONDS + 100.0)

    assert store.get("s1", now=SESSION_TTL_SECONDS + 100.0).pending_action is None
