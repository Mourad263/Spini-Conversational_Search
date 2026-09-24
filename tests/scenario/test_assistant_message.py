"""Task 3A: deterministic conversational response layer. Hermetic --
understand() stubbed, everything else real (same pattern as
test_orchestrator.py; no cross-test-module import, see that file)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.response import messages  # noqa: E402
from src.scenario.orchestrator import handle_message  # noqa: E402
from src.scenario.session_activity import SessionActivityStore  # noqa: E402
from src.search_adapter.adapter import FilterSet, SearchAdapter, SearchResult  # noqa: E402
from src.state_manager.state_manager import StateManager  # noqa: E402
from src.understanding.schema import Action, Facet, Intent  # noqa: E402


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


# --- unit-level: build_assistant_message() directly, no DB/network needed ---

def test_build_assistant_message_greeting_helper():
    # Sprint 6 social fast-path task: greeting now names the assistant, Spini.
    assert messages.greeting(False) == "Hello! I'm Spini. How can I help you?"
    assert messages.greeting(True) == "مرحباً! أنا Spini، كيف يمكنني مساعدتك؟"


def test_precedence_blocked_wins_over_everything():
    msg = messages.build_assistant_message(
        user_message="milk", intent="search", recommendation_request=True,
        recommendations_count=5, zero_result=True, likely_out_of_catalog=True, blocked=True,
    )
    assert msg == messages._ASSISTANT_MESSAGES["blocked"][0]


def test_precedence_zero_result_wins_over_recommendation_success():
    msg = messages.build_assistant_message(
        user_message="milk", intent="search", recommendation_request=True,
        recommendations_count=3, zero_result=True, likely_out_of_catalog=False, blocked=False,
    )
    assert msg == messages._ASSISTANT_MESSAGES["zero_result"][0]


def test_precedence_off_topic_never_mentions_products_even_with_stale_recommendation_flags():
    msg = messages.build_assistant_message(
        user_message="what's the weather?", intent="off_topic", recommendation_request=True,
        recommendations_count=5, zero_result=False, likely_out_of_catalog=False, blocked=False,
    )
    assert msg == messages._ASSISTANT_MESSAGES["off_topic"][0]


def test_recommendation_request_true_but_zero_recommendations_is_truthful():
    """Criterion 16: must NOT claim recommendations exist."""
    msg = messages.build_assistant_message(
        user_message="show me more", intent="no_op", recommendation_request=True,
        recommendations_count=0, zero_result=False, likely_out_of_catalog=False, blocked=False,
    )
    assert msg == messages._ASSISTANT_MESSAGES["recommendation_none"][0]
    assert "options you may like" not in msg


def test_recommendation_request_false_never_mentions_recommendations():
    """Criterion 17."""
    msg = messages.build_assistant_message(
        user_message="milk", intent="search", recommendation_request=False,
        recommendations_count=0, zero_result=False, likely_out_of_catalog=False, blocked=False,
    )
    assert msg == messages._ASSISTANT_MESSAGES["search_success"][0]


def test_search_and_recommendations_combined_message():
    msg = messages.build_assistant_message(
        user_message="what milk do you recommend?", intent="search", recommendation_request=True,
        recommendations_count=3, zero_result=False, likely_out_of_catalog=False, blocked=False,
    )
    assert msg == messages._ASSISTANT_MESSAGES["search_and_recommendations"][0]


def test_needs_clarification_takes_precedence_over_recommendation_none():
    msg = messages.build_assistant_message(
        user_message="what do you recommend?", intent="no_op", recommendation_request=True,
        recommendations_count=0, zero_result=False, likely_out_of_catalog=False, blocked=False,
        needs_clarification=True,
    )
    assert msg == messages._ASSISTANT_MESSAGES["recommendation_needs_clarification"][0]


def test_code_switch_with_arabic_script_selects_arabic():
    msg = messages.build_assistant_message(
        user_message="عايز milk من فضلك", intent="search", recommendation_request=False,
        recommendations_count=0, zero_result=False, likely_out_of_catalog=False, blocked=False,
    )
    assert msg == messages._ASSISTANT_MESSAGES["search_success"][1]


# --- end-to-end via handle_message() -- real validate()/StateManager/SearchAdapter ---

def test_english_search_success_end_to_end():
    # Sprint 6 recommendation-offer task: a successful normal search with
    # no recommendation_request of its own now appends a proactive offer.
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"))
    r = handle_message("t1", "I need milk", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=RecordingSearch())
    assert r.assistant_message == (
        "Sure, I found these products for you. Would you like me to recommend some alternatives?"
    )


def test_arabic_search_success_end_to_end():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(Action(intent=Intent.SEARCH, raw_query_text="جبنة", category_hint="جبنة"))
    r = handle_message("t2", "عايز جبنة", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=RecordingSearch())
    assert r.assistant_message == "أكيد، لقيت لك المنتجات دي. هل تريد أن أرشح لك بعض البدائل؟"


def test_search_plus_recommendations_end_to_end():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk", recommendation_request=True),
    )
    r = handle_message("t3", "what milk do you recommend?", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=RecordingSearch())
    assert len(r.recommendations) > 0
    assert r.assistant_message == "Sure, I found these products for you, and I also found some options you may like."


def test_recommendation_exploration_success_end_to_end():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"),
        Action(intent=Intent.NO_OP, recommendation_request=True),
    )
    handle_message("t4", "I need milk", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=RecordingSearch())
    r2 = handle_message("t4", "what do you recommend?", state_manager=sm, activity_store=store,
                         understand_fn=understand_fn, search_fn=RecordingSearch())
    assert len(r2.recommendations) > 0
    assert r2.assistant_message == "Here are some other options you may like."


def test_recommendation_request_zero_new_candidates_end_to_end():
    """All 3 real lactose-free milk products already shown -> honest
    'no additional options' wording, never a fallback to ordinary milk."""
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk", free_from="lactose",
               recommendation_request=True),
    )
    r = handle_message("t5", "what lactose-free milk do you recommend?", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=RecordingSearch())
    assert r.recommendations == []
    assert r.assistant_message == "You've already seen the available matching options."


def test_zero_search_result_end_to_end():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk", price_max=0.01),
    )
    r = handle_message("t6", "milk under 0.01", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=RecordingSearch())
    assert r.zero_result is True
    assert r.assistant_message == "I couldn't find products matching those filters."


def test_likely_out_of_catalog_end_to_end():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="zrqltnonexistentthing"),
    )
    r = handle_message("t7", "zrqltnonexistentthing", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=RecordingSearch())
    assert r.zero_result is True
    assert r.likely_out_of_catalog is True
    assert r.assistant_message == "I couldn't find a matching product in the current catalog."


def test_english_off_topic_end_to_end():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(Action(intent=Intent.OFF_TOPIC))
    r = handle_message("t8", "what's the weather?", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=RecordingSearch())
    assert r.assistant_message == "I can help you search for and explore products in the store."


def test_arabic_off_topic_end_to_end():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(Action(intent=Intent.OFF_TOPIC))
    r = handle_message("t9", "الجو عامل ايه النهاردة", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=RecordingSearch())
    assert r.assistant_message == "أقدر أساعدك تدور على المنتجات وتستكشف اختيارات من المتجر."


def test_reset_end_to_end():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"),
        Action(intent=Intent.RESET),
    )
    handle_message("t10", "milk", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=RecordingSearch())
    r2 = handle_message("t10", "start over", state_manager=sm, activity_store=store,
                         understand_fn=understand_fn, search_fn=RecordingSearch())
    assert r2.assistant_message == "Sure, I've cleared your current search. What would you like to find?"


def test_add_filter_end_to_end():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"),
        Action(intent=Intent.ADD_FILTER, brand_text="Juhayna"),
    )
    handle_message("t11", "milk", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=RecordingSearch())
    r2 = handle_message("t11", "also Juhayna", state_manager=sm, activity_store=store,
                         understand_fn=understand_fn, search_fn=RecordingSearch())
    assert r2.assistant_message == (
        "Sure, I've updated your search with that filter. Would you like me to recommend some alternatives?"
    )


def test_modify_filter_end_to_end():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"),
        Action(intent=Intent.MODIFY_FILTER, raw_query_text="cheese", category_hint="cheese"),
    )
    handle_message("t12", "milk", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=RecordingSearch())
    r2 = handle_message("t12", "cheese instead", state_manager=sm, activity_store=store,
                         understand_fn=understand_fn, search_fn=RecordingSearch())
    assert r2.assistant_message == "Sure, I've updated your search. Would you like me to recommend some alternatives?"


def test_remove_filter_end_to_end():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk", brand_text="Juhayna"),
        Action(intent=Intent.REMOVE_FILTER, target_facet=Facet.BRAND),
    )
    handle_message("t13", "Juhayna milk", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=RecordingSearch())
    r2 = handle_message("t13", "never mind the brand", state_manager=sm, activity_store=store,
                         understand_fn=understand_fn, search_fn=RecordingSearch())
    assert r2.assistant_message == "Sure, I've removed that filter. Would you like me to recommend some alternatives?"


def test_blocked_request_end_to_end():
    # REPEAT_THRESHOLD=4 (src/scenario/guard.py): the 1st/2nd/3rd identical
    # request all pass; only a 4th is blocked -- same pattern as
    # test_orchestrator.py's rapid-repeat test.
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"),
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"),
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"),
    )
    for t in (0.0, 1.0, 2.0):
        handle_message("t14", "milk please", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=RecordingSearch(), now=t)
    r4 = handle_message("t14", "milk please", state_manager=sm, activity_store=store,
                         understand_fn=understand_fn, search_fn=RecordingSearch(), now=3.0)
    assert r4.blocked is True
    assert r4.assistant_message == "I can't process that request. Please try a normal product search."


def test_existing_api_fields_remain_present():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"))
    r = handle_message("t15", "milk", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=RecordingSearch())
    for field in ("session_id", "message", "intent", "resolved_category", "resolved_brand",
                  "price_min", "price_max", "products", "total_count", "zero_result",
                  "likely_out_of_catalog", "blocked", "recommendations", "assistant_message"):
        assert hasattr(r, field)


def test_response_generation_does_not_mutate_state_manager():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk", price_max=50.0),
        Action(intent=Intent.NO_OP, recommendation_request=True),
    )
    handle_message("t16", "milk under 50", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=RecordingSearch())
    before = sm.get_state("t16")
    handle_message("t16", "what do you recommend?", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=RecordingSearch())
    assert sm.get_state("t16") == before
