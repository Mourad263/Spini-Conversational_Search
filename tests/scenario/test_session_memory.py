"""Sprint 6: session-scoped conversational memory (memory_recall). Hermetic
-- understand() always stubbed with a fixed/queued Action, same pattern as
test_orchestrator.py -- no real OpenAI/Groq calls. `search` is a
RecordingSearch so tests can assert recall NEVER reaches SearchAdapter
(read-only requirement), while normal turns still hit the real DB exactly
like every other orchestration test in this suite.
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


# --- A/B: previous-message and first-search recall ---

def test_previous_message_recall_returns_immediately_prior_turn():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"),
        Action(intent=Intent.ADD_FILTER, brand_text="Juhayna"),
        Action(intent=Intent.NO_OP, memory_recall="previous"),
    )
    search = RecordingSearch()
    handle_message("s1", "I need milk", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    handle_message("s1", "also Juhayna", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    calls_before_recall = len(search.calls)

    r = handle_message("s1", "what did I say before?", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert "also Juhayna" in r.assistant_message
    assert len(search.calls) == calls_before_recall  # never searched


def test_first_search_recall_returns_earliest_turn_not_the_latest():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"),
        Action(intent=Intent.ADD_FILTER, brand_text="Juhayna"),
        Action(intent=Intent.NO_OP, memory_recall="first"),
    )
    search = RecordingSearch()
    handle_message("s1", "I need milk", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    handle_message("s1", "also Juhayna", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)

    r = handle_message("s1", "what was the first thing I asked for?", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert "I need milk" in r.assistant_message
    assert "also Juhayna" not in r.assistant_message


# --- C: product/brand/price recall, sourced from canonical StateManager state ---

def test_product_recall_reflects_canonical_merged_state():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk", price_max=50.0),
        Action(intent=Intent.ADD_FILTER, brand_text="Juhayna"),
        Action(intent=Intent.NO_OP, memory_recall="product"),
    )
    search = RecordingSearch()
    handle_message("s1", "milk under 50", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    handle_message("s1", "also Juhayna", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)

    r = handle_message("s1", "what did I mention so far?", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert "milk_29" in r.assistant_message
    assert "juhayna" in r.assistant_message
    assert "50" in r.assistant_message


# --- D: Arabic recall ---

def test_arabic_recall_question_gets_arabic_answer():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="أرز", category_hint="rice", price_max=100.0),
        Action(intent=Intent.NO_OP, memory_recall="previous"),
    )
    search = RecordingSearch()
    handle_message("s1", "عايز أرز تحت مية", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)

    r = handle_message("s1", "قولت إيه قبل كده؟", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert "عايز أرز تحت مية" in r.assistant_message
    assert "Earlier you said" not in r.assistant_message


# --- E: session isolation ---

def test_recall_is_isolated_per_session():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"),
        Action(intent=Intent.NO_OP, memory_recall="previous"),
    )
    search = RecordingSearch()
    handle_message("session-a", "I need milk", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)

    r = handle_message("session-b", "what did I say before?", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert "milk" not in r.assistant_message
    assert "I need milk" not in r.assistant_message


# --- F: recall is read-only -- canonical state and history byte-for-byte unchanged ---

def test_recall_never_mutates_state_or_history():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk", price_max=50.0),
        Action(intent=Intent.NO_OP, memory_recall="product"),
    )
    search = RecordingSearch()
    handle_message("s1", "milk under 50", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)

    state_before = sm.get_state("s1")
    history_before = list(store.get("s1").history)

    handle_message("s1", "what did I mention?", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)

    assert sm.get_state("s1") == state_before
    # the recall turn itself DOES get appended (same NO_OP contract as any
    # other conversational turn) -- but the PRIOR entries are untouched.
    assert store.get("s1").history[:len(history_before)] == history_before


# --- G: reset clears memory too ---

def test_reset_preserves_memory_recall_history():
    """Sprint 6 lifecycle task: a normal "start over" only clears the
    active shopping search -- it must NOT erase prior conversation turns.
    (Superseded scope note: this test previously asserted the opposite,
    full-wipe-on-reset behavior; see tests/scenario/test_session_lifecycle.py
    for the full reset-vs-expiration lifecycle contract.)"""
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"),
        Action(intent=Intent.RESET),
        Action(intent=Intent.NO_OP, memory_recall="first"),
    )
    search = RecordingSearch()
    handle_message("s1", "I need milk", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    handle_message("s1", "start over", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)

    r = handle_message("s1", "what was the first thing I asked?", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert "I need milk" in r.assistant_message


# --- H: honest "nothing yet" when there is genuinely no memory ---

def test_no_memory_yet_gives_honest_fallback_not_a_guess():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(Action(intent=Intent.NO_OP, memory_recall="previous"))
    search = RecordingSearch()

    r = handle_message("brand-new-session", "what did I say before?", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert len(search.calls) == 0
    assert r.products == []
    assert r.assistant_message  # non-empty, always populated
    assert "don't have" in r.assistant_message.lower()


def test_no_memory_yet_product_recall_honest_fallback():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(Action(intent=Intent.NO_OP, memory_recall="product"))
    search = RecordingSearch()

    r = handle_message("brand-new-session-2", "what product did I mention?", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert "haven't mentioned" in r.assistant_message


# --- I: normal search/filter turns are unaffected by memory_recall existing ---

def test_normal_search_regression_unaffected_by_memory_field():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk", price_max=50.0),
    )
    search = RecordingSearch()
    r = handle_message("s1", "milk under 50", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert r.resolved_category == "milk_29"
    assert r.price_max == 50.0
    assert len(search.calls) == 1


# --- J: a recall turn interleaved with recommendation exploration doesn't break it ---

def test_recall_interleaved_with_recommendation_flow_still_works():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"),
        Action(intent=Intent.NO_OP, memory_recall="previous"),
        Action(intent=Intent.NO_OP, recommendation_request=True),
    )
    search = RecordingSearch()
    handle_message("s1", "I need milk", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    r_recall = handle_message("s1", "what did I say before?", state_manager=sm, activity_store=store,
                               understand_fn=understand_fn, search_fn=search)
    assert "I need milk" in r_recall.assistant_message
    calls_after_recall = len(search.calls)

    r_rec = handle_message("s1", "what do you recommend?", state_manager=sm, activity_store=store,
                            understand_fn=understand_fn, search_fn=search)
    # Pure continuation, nothing new explored -- recommendation evidence
    # reuses last_shown_products from the "milk" search instead of
    # re-searching, so no new SearchAdapter call happens here.
    assert len(search.calls) == calls_after_recall
    assert r_rec.assistant_message
