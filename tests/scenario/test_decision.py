"""Sprint 4 Part 2: deterministic Action-invariant and recommendation-
policy unit tests. Hermetic -- hand-built Action/FilterSet fixtures, no
LLM, no DB (same style as tests/state_manager/test_state_manager.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest  # noqa: E402

from src.scenario.decision import (  # noqa: E402
    MAX_RELAXATION_STEPS,
    RELAXATION_FACTOR,
    derive_recommendation_search,
    has_searchable_target,
    normalize_recommendation_intent,
    normalize_target_facet,
)
from src.scenario.session_activity import SessionActivity  # noqa: E402
from src.search_adapter.adapter import FilterSet  # noqa: E402
from src.understanding.schema import Action, Facet, Intent  # noqa: E402


# --- target_facet invariant ---

def test_non_remove_filter_with_accidental_target_facet_is_normalized_to_none():
    action = Action(intent=Intent.MODIFY_FILTER, price_min=50.0, target_facet=Facet.PRICE)
    result = normalize_target_facet(action)
    assert result.target_facet is None
    assert result.intent == Intent.MODIFY_FILTER
    assert result.price_min == 50.0


def test_remove_filter_preserves_a_valid_target_facet():
    action = Action(intent=Intent.REMOVE_FILTER, target_facet=Facet.BRAND)
    result = normalize_target_facet(action)
    assert result.target_facet == Facet.BRAND


def test_normalize_target_facet_does_not_touch_other_fields():
    action = Action(
        intent=Intent.SEARCH, price_min=10.0, price_max=50.0,
        raw_query_text="milk", target_facet=Facet.CATEGORY,
        recommendation_request=True, customer_service_tone=True,
    )
    result = normalize_target_facet(action)
    assert result.target_facet is None
    assert result.intent == Intent.SEARCH
    assert result.price_min == 10.0
    assert result.price_max == 50.0
    assert result.raw_query_text == "milk"
    assert result.recommendation_request is True
    assert result.customer_service_tone is True


def test_normalize_target_facet_is_a_noop_when_already_none():
    action = Action(intent=Intent.SEARCH, raw_query_text="milk")
    result = normalize_target_facet(action)
    assert result.target_facet is None


# --- searchable target ---

@pytest.mark.parametrize("filters,expected", [
    (FilterSet(), False),
    (FilterSet(price_max=100), False),
    (FilterSet(query_text="lotion"), True),
    (FilterSet(category="lotion_1"), True),
    (FilterSet(brand="nivea"), True),
])
def test_has_searchable_target(filters, expected):
    assert has_searchable_target(filters) == expected


# --- recommendation intent normalization (normalize_recommendation_intent) ---
# ONE function, two mutually-exclusive directions (action.intent can never be
# both NO_OP and a state-changing intent at once): direction 1 repairs a
# NO_OP that should have been a real intent because a price WAS stated;
# direction 2 (Sprint 6 recommendation-context task) repairs a state-
# changing intent that should have been NO_OP because NOTHING was stated.

def test_repair_empty_state_becomes_search():
    action = Action(intent=Intent.NO_OP, recommendation_request=True, price_max=100.0)
    result = normalize_recommendation_intent(action, FilterSet())
    assert result.intent == Intent.SEARCH
    assert result.price_max == 100.0
    assert result.recommendation_request is True


def test_repair_existing_target_no_price_becomes_add_filter():
    action = Action(intent=Intent.NO_OP, recommendation_request=True, price_max=100.0)
    current = FilterSet(category="lotion_1")
    result = normalize_recommendation_intent(action, current)
    assert result.intent == Intent.ADD_FILTER
    assert result.price_max == 100.0


def test_repair_existing_target_with_price_becomes_modify_filter():
    action = Action(intent=Intent.NO_OP, recommendation_request=True, price_max=100.0)
    current = FilterSet(category="lotion_1", price_max=150.0)
    result = normalize_recommendation_intent(action, current)
    assert result.intent == Intent.MODIFY_FILTER
    assert result.price_max == 100.0


def test_repair_does_not_fire_for_bare_recommendation():
    action = Action(intent=Intent.NO_OP, recommendation_request=True)
    result = normalize_recommendation_intent(
        action, FilterSet(category="lotion_1", price_max=100.0)
    )
    assert result.intent == Intent.NO_OP


def test_repair_does_not_fire_for_show_me_more_with_no_new_price():
    action = Action(intent=Intent.NO_OP, recommendation_request=True, raw_query_text=None)
    result = normalize_recommendation_intent(
        action, FilterSet(category="lotion_1", price_max=100.0)
    )
    assert result.intent == Intent.NO_OP


def test_repair_does_not_fire_for_non_no_op_intents_that_already_state_a_value():
    action = Action(intent=Intent.SEARCH, price_max=100.0)
    result = normalize_recommendation_intent(action, FilterSet())
    assert result.intent == Intent.SEARCH


def test_repair_does_not_fire_without_recommendation_request():
    action = Action(intent=Intent.NO_OP, recommendation_request=False, price_max=100.0)
    result = normalize_recommendation_intent(action, FilterSet())
    assert result.intent == Intent.NO_OP


# --- direction 2: an empty state-changing recommendation request becomes NO_OP ---

def test_empty_search_with_recommendation_request_becomes_no_op():
    # the real live-confirmed shape for "recommend me more"
    action = Action(intent=Intent.SEARCH, recommendation_request=True)
    result = normalize_recommendation_intent(action, FilterSet(category="chips_1", price_min=20.0))
    assert result.intent == Intent.NO_OP
    assert result.recommendation_request is True


def test_empty_add_filter_and_modify_filter_with_recommendation_request_become_no_op():
    for intent in (Intent.ADD_FILTER, Intent.MODIFY_FILTER):
        action = Action(intent=intent, recommendation_request=True)
        result = normalize_recommendation_intent(action, FilterSet())
        assert result.intent == Intent.NO_OP


def test_search_with_new_query_and_recommendation_request_is_not_forced_to_no_op():
    action = Action(intent=Intent.SEARCH, raw_query_text="chocolate", recommendation_request=True)
    result = normalize_recommendation_intent(action, FilterSet(query_text="chips"))
    assert result.intent == Intent.SEARCH
    assert result.raw_query_text == "chocolate"


def test_search_with_new_category_or_brand_and_recommendation_request_is_not_forced_to_no_op():
    action = Action(intent=Intent.SEARCH, category_hint="chocolate", recommendation_request=True)
    result = normalize_recommendation_intent(action, FilterSet(query_text="chips"))
    assert result.intent == Intent.SEARCH

    action2 = Action(intent=Intent.ADD_FILTER, brand_text="lindt", recommendation_request=True)
    result2 = normalize_recommendation_intent(action2, FilterSet(query_text="chocolate"))
    assert result2.intent == Intent.ADD_FILTER


def test_search_with_new_price_and_recommendation_request_is_not_forced_to_no_op():
    action = Action(intent=Intent.SEARCH, price_max=100.0, recommendation_request=True)
    result = normalize_recommendation_intent(action, FilterSet(query_text="chips"))
    assert result.intent == Intent.SEARCH
    assert result.price_max == 100.0


def test_empty_search_without_recommendation_request_is_untouched():
    action = Action(intent=Intent.SEARCH)
    result = normalize_recommendation_intent(action, FilterSet(query_text="chips"))
    assert result.intent == Intent.SEARCH


def test_ordinary_search_with_a_stated_value_and_no_recommendation_request_is_unchanged():
    action = Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk")
    result = normalize_recommendation_intent(action, FilterSet())
    assert result is action  # untouched, same object -- not even copied


# --- recommendation relaxation (Sprint 4 Part 2 micro-correction: each step
# explores the NEXT price BAND above the previous ceiling, not a widened
# 0..relaxed_max range that would just re-return already-seen products) ---

def test_relaxation_band_step_1_2_3_from_canonical_max_100():
    activity = SessionActivity()
    current = FilterSet(category="lotion_1", price_max=100.0)

    outcome1 = derive_recommendation_search(current, activity)
    assert outcome1.effective_filters.price_min == 100.0
    assert outcome1.effective_filters.price_max == 120.0
    assert outcome1.relaxation_step == 1
    assert outcome1.relaxation_applied is True

    outcome2 = derive_recommendation_search(current, activity)
    assert outcome2.effective_filters.price_min == 120.0
    assert outcome2.effective_filters.price_max == 144.0
    assert outcome2.relaxation_step == 2

    outcome3 = derive_recommendation_search(current, activity)
    assert outcome3.effective_filters.price_min == 144.0
    assert outcome3.effective_filters.price_max == 172.8
    assert outcome3.relaxation_step == 3

    # step 4: relaxation exhausted
    outcome4 = derive_recommendation_search(current, activity)
    assert outcome4.relaxation_exhausted is True
    assert outcome4.effective_filters is None


def test_relaxation_exhausted_after_max_steps():
    activity = SessionActivity(relaxation_step=MAX_RELAXATION_STEPS)
    current = FilterSet(category="lotion_1", price_max=100.0)

    outcome = derive_recommendation_search(current, activity)
    assert outcome.relaxation_exhausted is True
    assert outcome.effective_filters is None
    assert outcome.relaxation_applied is False
    assert activity.relaxation_step == MAX_RELAXATION_STEPS  # unchanged


def test_relaxation_factor_constants():
    assert RELAXATION_FACTOR == 1.20
    assert MAX_RELAXATION_STEPS == 3


def test_relaxation_band_preserves_query_category_brand():
    activity = SessionActivity()
    current = FilterSet(category="lotion_1", brand="nivea", query_text="lotion", price_max=100.0)

    outcome = derive_recommendation_search(current, activity)

    assert outcome.effective_filters.category == "lotion_1"
    assert outcome.effective_filters.brand == "nivea"
    assert outcome.effective_filters.query_text == "lotion"


def test_relaxation_band_ignores_canonical_price_min_for_the_band_floor():
    """Canonical price_min=50 must NOT survive into the band's lower edge --
    the band always starts from the canonical price_max, per the approved
    product behavior (band 1 = [max, max*1.20])."""
    activity = SessionActivity()
    current = FilterSet(category="lotion_1", price_min=50.0, price_max=100.0)

    outcome = derive_recommendation_search(current, activity)

    assert outcome.effective_filters.price_min == 100.0
    assert outcome.effective_filters.price_max == 120.0


def test_canonical_state_never_mutated_by_relaxation():
    activity = SessionActivity()
    current = FilterSet(category="lotion_1", price_min=50.0, price_max=100.0)

    outcome = derive_recommendation_search(current, activity)

    assert current.price_max == 100.0  # untouched
    assert current.price_min == 50.0  # untouched
    assert current.category == "lotion_1"
    assert outcome.effective_filters is not current  # a separate object, never the
    # canonical instance itself


def test_explicit_user_price_change_is_not_a_relaxation_call():
    """An explicit new user-stated price (e.g. "show me lotion under 150")
    is normal state-transition handling, not a recommendation/exploration
    turn -- Part 3 orchestration resets relaxation_step via
    SessionActivityStore.reset_relaxation_step() and never calls
    derive_recommendation_search() for that same turn, so no automatic
    1.20 band is layered on top of a value the user actually stated."""
    from src.scenario.session_activity import SessionActivityStore

    store = SessionActivityStore()
    session = "s1"
    activity = store.get(session)
    activity.relaxation_step = 2  # from prior "show me more" exploration turns

    # the real orchestration action for an explicit price change: reset the step,
    # and derive_recommendation_search is simply never invoked for this turn.
    store.reset_relaxation_step(session)

    assert store.get(session).relaxation_step == 0


def test_canonical_target_with_no_price_used_as_is_no_price_invented():
    activity = SessionActivity()
    current = FilterSet(category="lotion_1")

    outcome = derive_recommendation_search(current, activity)

    assert outcome.needs_clarification is False
    assert outcome.effective_filters.category == "lotion_1"
    assert outcome.effective_filters.price_max is None
    assert outcome.relaxation_applied is False
    assert activity.relaxation_step == 0


def test_empty_state_needs_clarification_no_search_target():
    activity = SessionActivity()
    outcome = derive_recommendation_search(FilterSet(), activity)

    assert outcome.needs_clarification is True
    assert outcome.effective_filters is None


def test_price_only_state_also_needs_clarification():
    activity = SessionActivity()
    outcome = derive_recommendation_search(FilterSet(price_max=100.0), activity)

    assert outcome.needs_clarification is True
    assert outcome.effective_filters is None
