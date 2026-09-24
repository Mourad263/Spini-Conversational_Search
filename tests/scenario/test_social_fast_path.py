"""Sprint 6 social fast-path: greeting/how-are-you/gratitude answered
deterministically before understand() ever runs (src/scenario/social.py +
src/scenario/orchestrator.py). Hermetic -- `understand_fn` here RAISES if
called at all, proving a social turn never reaches the LLM (stronger than
just checking trace.llm_calls==0). Mixed-content tests use a normal stub
to prove the existing workflow still runs. Trace is read back via caplog,
same pattern as tests/scenario/test_request_trace.py.
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
    raise AssertionError(f"understand() was called for a social-only message: {message!r}")


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


# --- A: English greeting ---

def test_english_greeting_fast_path(caplog):
    sm, store = StateManager(), SessionActivityStore()
    search = RecordingSearch()
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        r = handle_message("s1", "hello", state_manager=sm, activity_store=store,
                            understand_fn=understand_must_not_be_called, search_fn=search)

    assert r.assistant_message == "Hello! I'm Spini. How can I help you?"
    assert r.products == [] and r.recommendations == []
    trace = _last_trace(caplog)
    assert trace["route"] == "social_fast_path"
    assert trace["llm_calls"] == 0
    assert trace["search_calls"] == 0
    assert trace["recommendation_calls"] == 0
    assert trace["validation_calls"] == 0
    assert trace["embedding_calls"] == 0
    assert len(search.calls) == 0


# --- B: Arabic greeting ---

def test_arabic_greeting_fast_path(caplog):
    sm, store = StateManager(), SessionActivityStore()
    search = RecordingSearch()
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        r = handle_message("s1", "مرحبا", state_manager=sm, activity_store=store,
                            understand_fn=understand_must_not_be_called, search_fn=search)

    assert r.assistant_message == "مرحباً! أنا Spini، كيف يمكنني مساعدتك؟"
    trace = _last_trace(caplog)
    assert trace["route"] == "social_fast_path"
    assert trace["llm_calls"] == 0


# --- C: Egyptian how-are-you ---

def test_egyptian_how_are_you_fast_path():
    sm, store = StateManager(), SessionActivityStore()
    search = RecordingSearch()
    r = handle_message("s1", "ازيك", state_manager=sm, activity_store=store,
                        understand_fn=understand_must_not_be_called, search_fn=search)

    assert r.assistant_message == "بخير، كيف يمكنني مساعدتك؟"
    assert len(search.calls) == 0


# --- D: English gratitude at the beginning of a conversation ---

def test_early_gratitude_response():
    sm, store = StateManager(), SessionActivityStore()
    search = RecordingSearch()
    r = handle_message("s1", "thanks", state_manager=sm, activity_store=store,
                        understand_fn=understand_must_not_be_called, search_fn=search)

    assert r.assistant_message == "You're welcome! How can I help you?"


# --- E: Arabic gratitude after real shopping activity ---

def test_late_gratitude_response_after_shopping_activity():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk", price_max=50.0),
    )
    search = RecordingSearch()
    handle_message("s1", "I need milk under 50", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    state_before = sm.get_state("s1")

    r = handle_message("s1", "شكراً", state_manager=sm, activity_store=store,
                        understand_fn=understand_must_not_be_called, search_fn=search)

    assert r.assistant_message == "العفو! أتمنى أكون قدرت أساعدك."
    assert sm.get_state("s1") == state_before  # shopping state unchanged


# --- K: name question, English ---

def test_name_question_english_fast_path():
    sm, store = StateManager(), SessionActivityStore()
    search = RecordingSearch()
    r = handle_message("s1", "what's your name", state_manager=sm, activity_store=store,
                        understand_fn=understand_must_not_be_called, search_fn=search)

    assert r.assistant_message == "My name is Spini."
    assert len(search.calls) == 0


def test_name_question_english_variant_with_punctuation():
    sm, store = StateManager(), SessionActivityStore()
    search = RecordingSearch()
    r = handle_message("s1", "What is your name?", state_manager=sm, activity_store=store,
                        understand_fn=understand_must_not_be_called, search_fn=search)

    assert r.assistant_message == "My name is Spini."


# --- L: name question, Arabic/Egyptian ---

def test_name_question_arabic_fast_path():
    sm, store = StateManager(), SessionActivityStore()
    search = RecordingSearch()
    r = handle_message("s1", "اسمك ايه", state_manager=sm, activity_store=store,
                        understand_fn=understand_must_not_be_called, search_fn=search)

    assert r.assistant_message == "اسمي Spini."


def test_name_question_arabic_variant():
    sm, store = StateManager(), SessionActivityStore()
    search = RecordingSearch()
    r = handle_message("s1", "ما اسمك؟", state_manager=sm, activity_store=store,
                        understand_fn=understand_must_not_be_called, search_fn=search)

    assert r.assistant_message == "اسمي Spini."


# --- M: mixed name question + shopping is NOT intercepted ---

def test_mixed_name_question_and_shopping_uses_real_workflow():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"),
    )
    search = RecordingSearch()
    r = handle_message("s1", "what's your name and show me milk", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert r.resolved_category == "milk_29"
    assert len(search.calls) == 1


# --- N: name question trace has zero expensive calls ---

def test_name_question_trace_has_zero_expensive_calls(caplog):
    sm, store = StateManager(), SessionActivityStore()
    search = RecordingSearch()
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        handle_message("s1", "who are you", state_manager=sm, activity_store=store,
                        understand_fn=understand_must_not_be_called, search_fn=search)

    trace = _last_trace(caplog)
    assert trace["route"] == "social_fast_path"
    assert trace["llm_calls"] == 0
    assert trace["search_calls"] == 0
    assert trace["recommendation_calls"] == 0
    assert trace["embedding_calls"] == 0


# --- F: mixed greeting + shopping is NOT intercepted ---

def test_mixed_greeting_and_shopping_uses_real_workflow():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"),
    )
    search = RecordingSearch()
    r = handle_message("s1", "hello, I need milk", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert r.resolved_category == "milk_29"
    assert len(search.calls) == 1


# --- G: mixed gratitude + action is NOT intercepted ---

def test_mixed_gratitude_and_action_uses_real_workflow():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"),
        Action(intent=Intent.NO_OP, recommendation_request=True),
    )
    search = RecordingSearch()
    handle_message("s1", "I need milk", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)

    r = handle_message("s1", "thanks, show me more", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert r.assistant_message != "You're welcome! How can I help you?"
    assert len(search.calls) >= 1


# --- H: uncertain casual/shop-adjacent message falls through ---

def test_uncertain_casual_message_falls_through_to_real_workflow():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(Action(intent=Intent.OFF_TOPIC))
    search = RecordingSearch()
    r = handle_message("s1", "hi, do you have rice?", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert r.intent == "off_topic"  # reached the real workflow, not the fast path


# --- I: memory/state regression -- social turn doesn't erase history/memory ---

def test_social_turn_does_not_erase_prior_memory_or_state():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"),
        Action(intent=Intent.NO_OP, memory_recall="first"),
    )
    search = RecordingSearch()
    handle_message("s1", "I need milk", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    history_before = list(store.get("s1").history)
    state_before = sm.get_state("s1")

    handle_message("s1", "thank you", state_manager=sm, activity_store=store,
                    understand_fn=understand_must_not_be_called, search_fn=search)

    assert store.get("s1").history == history_before
    assert sm.get_state("s1") == state_before

    r = handle_message("s1", "what was the first thing I asked?", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)
    assert "I need milk" in r.assistant_message


# --- J: exactly one trace, zero expensive calls, for a pure social turn ---

def test_social_trace_has_zero_expensive_calls_and_measures_latency(caplog):
    sm, store = StateManager(), SessionActivityStore()
    search = RecordingSearch()
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        handle_message("s1", "hello", state_manager=sm, activity_store=store,
                        understand_fn=understand_must_not_be_called, search_fn=search)

    records = [r for r in caplog.records if r.name == LOGGER_NAME and r.msg == "request_trace %s"]
    assert len(records) == 1
    trace = records[0].args
    assert trace["llm_calls"] == 0
    assert trace["llm_input_tokens"] is None
    assert trace["llm_output_tokens"] is None
    assert trace["search_calls"] == 0
    assert trace["recommendation_calls"] == 0
    assert trace["embedding_calls"] == 0
    assert isinstance(trace["total_latency_ms"], float) and trace["total_latency_ms"] >= 0


# --- O: minor spelling typo of a known greeting -- fuzzy fast path ---

def test_minor_typo_greeting_fast_path(caplog):
    sm, store = StateManager(), SessionActivityStore()
    search = RecordingSearch()
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        r = handle_message("s1", "hallo", state_manager=sm, activity_store=store,
                            understand_fn=understand_must_not_be_called, search_fn=search)

    assert r.assistant_message == "Hello! I'm Spini. How can I help you?"
    trace = _last_trace(caplog)
    assert trace["route"] == "social_fast_path"
    assert trace["llm_calls"] == 0
    assert trace["validation_calls"] == 0
    assert trace["search_calls"] == 0
    assert trace["recommendation_calls"] == 0
    assert trace["embedding_calls"] == 0


# --- P: casual "whats up" -- same how-are-you response family ---

def test_casual_whats_up_fast_path(caplog):
    sm, store = StateManager(), SessionActivityStore()
    search = RecordingSearch()
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        r = handle_message("s1", "whats up", state_manager=sm, activity_store=store,
                            understand_fn=understand_must_not_be_called, search_fn=search)

    assert r.assistant_message == "I'm doing well, thanks for asking! How can I help you?"
    trace = _last_trace(caplog)
    assert trace["route"] == "social_fast_path"
    assert trace["llm_calls"] == 0


# --- Q: contraction/punctuation/case variants of "whats up" -- same semantics ---

def test_whats_up_contraction_and_punctuation_variants():
    sm, store = StateManager(), SessionActivityStore()
    search = RecordingSearch()
    for text in ("what's up", "what’s up", "WHAT'S UP", "what's up?"):
        r = handle_message("s1", text, state_manager=sm, activity_store=store,
                            understand_fn=understand_must_not_be_called, search_fn=search)
        assert r.assistant_message == "I'm doing well, thanks for asking! How can I help you?"


# --- Q2: apostrophe normalization regression -- "how's it going" (its own
# apostrophe form was removed from the vocabulary; now covered by the
# general normalization step instead of a duplicate literal entry) ---

def test_hows_it_going_apostrophe_form_still_works():
    sm, store = StateManager(), SessionActivityStore()
    r = handle_message("s1", "how's it going", state_manager=sm, activity_store=store,
                        understand_fn=understand_must_not_be_called, search_fn=RecordingSearch())
    assert r.assistant_message == "I'm doing well, thanks for asking! How can I help you?"


# --- R: minor-typo generalization -- two more canonical forms besides "hello" ---

def test_minor_typo_generalization_controls():
    sm, store = StateManager(), SessionActivityStore()
    search = RecordingSearch()

    r1 = handle_message("s1", "helo", state_manager=sm, activity_store=store,
                         understand_fn=understand_must_not_be_called, search_fn=search)
    assert r1.assistant_message == "Hello! I'm Spini. How can I help you?"

    r2 = handle_message("s1", "thnaks", state_manager=sm, activity_store=store,
                         understand_fn=understand_must_not_be_called, search_fn=search)
    assert r2.assistant_message == "You're welcome! How can I help you?"


# --- S: an unrelated word that merely LOOKS similar must not become social ---

def test_unrelated_near_match_is_not_falsely_social():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(Action(intent=Intent.OFF_TOPIC))
    search = RecordingSearch()

    r = handle_message("s1", "yellow", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)
    assert r.intent == "off_topic"  # reached the real workflow, not the fast path


# --- S2: "cello"/"jello" -- real words at the SAME rapidfuzz ratio (80.0) as
# the genuine "hallo" typo against "hello", but with a different first
# letter -- exactly the disclosed collision this closure fixes ---

def test_cello_jello_same_edit_distance_as_hallo_stay_non_social():
    sm, store = StateManager(), SessionActivityStore()
    for word in ("cello", "jello"):
        understand_fn = stub_understand(Action(intent=Intent.OFF_TOPIC))
        r = handle_message("s1", word, state_manager=sm, activity_store=store,
                            understand_fn=understand_fn, search_fn=RecordingSearch())
        assert r.intent == "off_topic"


# --- S3: independently-selected same-distance near-collision for another
# canonical form (gratitude: "thanks") -- "shanks" sits at ratio()=83.3,
# the SAME score as the genuine "thnaks" typo, proving the first-letter
# rule (not just a low score) is what rejects it -- generalizes the fix ---

def test_gratitude_near_collision_word_stays_non_social():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(Action(intent=Intent.OFF_TOPIC))
    r = handle_message("s1", "shanks", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=RecordingSearch())
    assert r.intent == "off_topic"


# --- S4: same for the how-are-you form -- "chats up" sits at ratio()=87.5
# against "whats up" (higher than the genuine "hallo" typo score) but with
# a different first letter ---

def test_how_are_you_near_collision_phrase_stays_non_social():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(Action(intent=Intent.OFF_TOPIC))
    r = handle_message("s1", "chats up", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=RecordingSearch())
    assert r.intent == "off_topic"


# --- T: mixed casual-greeting + shopping content is NOT intercepted ---

def test_mixed_whats_up_and_shopping_uses_real_workflow():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="pepsi", brand_text="pepsi"),
    )
    search = RecordingSearch()
    r = handle_message("s1", "whats up with pepsi prices", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert r.resolved_brand == "pepsi"
    assert len(search.calls) == 1


def test_mixed_hey_and_shopping_uses_real_workflow():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="cheese", category_hint="cheese"),
    )
    search = RecordingSearch()
    r = handle_message("s1", "hey i need cheese", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert len(search.calls) == 1
