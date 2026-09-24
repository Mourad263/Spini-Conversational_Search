"""Sprint 6 observability task: per-message RequestTrace (src/scenario/
trace.py). Hermetic -- understand() always stubbed; the emitted trace is
read back via caplog (the log record's raw args, not a string re-parse),
since tracing is deliberately server-side-only and never returned in
SessionMessageResponse. RecordingSearch wraps the REAL SearchAdapter, same
pattern as test_orchestrator.py -- real DB, no LLM/network for understand().
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


def stub_understand(*actions_or_exc):
    it = iter(actions_or_exc)

    def _stub(message: str, history: list[str] | None = None) -> Action:
        item = next(it)
        if isinstance(item, Exception):
            raise item
        return item

    return _stub


class RecordingSearch:
    def __init__(self) -> None:
        self.calls: list[FilterSet] = []

    def __call__(self, filters: FilterSet, limit: int, offset: int) -> SearchResult:
        self.calls.append(filters)
        return SearchAdapter().search(filters, limit=limit, offset=offset)


def _last_trace(caplog) -> dict:
    records = [r for r in caplog.records if r.name == LOGGER_NAME and r.msg == "request_trace %s"]
    assert records, "expected exactly one request_trace log record"
    # logging's %-formatting special-cases a single Mapping arg: record.args
    # IS the dict itself here, not a 1-tuple containing it.
    return records[-1].args


# --- A: normal search path ---

def test_search_trace_counts_llm_validation_search(caplog):
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"),
    )
    search = RecordingSearch()
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        r = handle_message("s1", "I need milk", state_manager=sm, activity_store=store,
                            understand_fn=understand_fn, search_fn=search)

    trace = _last_trace(caplog)
    assert trace["session_id"] == "s1"
    assert trace["route"] == "shopping_search"
    assert trace["llm_calls"] == 1
    assert trace["validation_calls"] == 1
    assert trace["search_calls"] == 1
    assert trace["recommendation_calls"] == 0
    assert isinstance(trace["total_latency_ms"], float) and trace["total_latency_ms"] >= 0
    # stub understand() never sets real usage -- must stay null, not fabricated
    assert trace["llm_input_tokens"] is None
    assert r.resolved_category == "milk_29"  # behavior itself unaffected


# --- B: memory recall -- no search, no recommendation ---

def test_memory_recall_trace_has_zero_search_and_recommendation_calls(caplog):
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(Action(intent=Intent.NO_OP, memory_recall="first"))
    search = RecordingSearch()
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        handle_message("s1", "what did I ask for first?", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    trace = _last_trace(caplog)
    assert trace["route"] == "memory_recall"
    assert trace["llm_calls"] == 1
    assert trace["search_calls"] == 0
    assert trace["recommendation_calls"] == 0
    assert trace["validation_calls"] == 0
    assert len(search.calls) == 0


# --- C: recommendation path reflects actual invocation ---

def test_recommendation_trace_reflects_actual_invocation(caplog):
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"),
        Action(intent=Intent.NO_OP, recommendation_request=True),
    )
    search = RecordingSearch()
    handle_message("s1", "milk", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        handle_message("s1", "what do you recommend?", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    trace = _last_trace(caplog)
    assert trace["route"] == "recommendation"
    assert trace["recommendation_calls"] == 1
    # Pure continuation (no price band to explore) reuses last_shown_products
    # from turn 1's own search instead of re-searching -- zero new calls.
    assert trace["search_calls"] == 0


# --- C2: the exact reported browser sequence (real typo repair collapses
# to the same repaired query on both turns; a NO_OP continuation follows) ---

def test_pepsi_typo_then_pure_continuation_makes_no_redundant_search(caplog):
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="pepsi", brand_text="pepsi"),
        Action(intent=Intent.SEARCH, raw_query_text="pepsi", brand_text="pepsi"),
        Action(intent=Intent.NO_OP, recommendation_request=True),
    )
    search = RecordingSearch()
    handle_message("s1", "pepsi", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    handle_message("s1", "pebsi", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        handle_message("s1", "suggest to me more", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    trace = _last_trace(caplog)
    assert trace["route"] == "recommendation"
    assert trace["recommendation_calls"] == 1
    assert trace["search_calls"] == 0


def test_no_recommendation_means_zero_recommendation_calls(caplog):
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"),
    )
    search = RecordingSearch()
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        handle_message("s1", "I need milk", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    trace = _last_trace(caplog)
    assert trace["recommendation_calls"] == 0


# --- D: provider error -- trace still completes, no fake token values ---

def test_provider_error_trace_completes_without_fabricated_tokens(caplog):
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(RuntimeError("simulated provider failure"))
    search = RecordingSearch()

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        r = handle_message("s1", "I need milk", state_manager=sm, activity_store=store,
                            understand_fn=understand_fn, search_fn=search)

    trace = _last_trace(caplog)
    assert trace["route"] == "error"
    assert trace["llm_calls"] == 1  # the call was attempted
    assert trace["llm_input_tokens"] is None
    assert trace["llm_output_tokens"] is None
    assert trace["llm_total_tokens"] is None
    assert trace["search_calls"] == 0
    assert r.intent == "error"  # normal error response unaffected


# --- E: no customer-facing behavior change from tracing ---

def test_response_shape_has_no_trace_fields():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"))
    search = RecordingSearch()
    r = handle_message("s1", "I need milk", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    dumped = r.model_dump()
    for forbidden in ("trace", "llm_calls", "search_calls", "recommendation_calls", "latency_ms"):
        assert forbidden not in dumped


# --- one trace per request, even across two sequential turns ---

def test_exactly_one_trace_per_request(caplog):
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"),
        Action(intent=Intent.OFF_TOPIC),
    )
    search = RecordingSearch()
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        handle_message("s1", "I need milk", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)
        handle_message("s1", "what's the weather?", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    records = [r for r in caplog.records if r.name == LOGGER_NAME and r.msg == "request_trace %s"]
    assert len(records) == 2
    assert records[0].args["route"] == "shopping_search"
    assert records[1].args["route"] == "off_topic"
