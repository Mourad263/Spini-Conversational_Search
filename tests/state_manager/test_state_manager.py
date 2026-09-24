"""Sprint 3: State Manager (tasks.md T027/T033/T034) unit tests.

Fast and hermetic on purpose -- no LLM call, no Postgres. Feeds
StateManager.apply() hand-built (Action, resolved FilterSet) pairs
directly and asserts the resulting per-session state, covering every
intent's update rule in isolation: search/reset replace, add_filter/
modify_filter merge a field, remove_filter clears exactly the field named
by target_facet. The real multi-turn, real-LLM version of this same
component lives in tests/understanding/test_state_manager_e2e.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest  # noqa: E402

from src.search_adapter.adapter import FilterSet, InvalidFilterError  # noqa: E402
from src.state_manager.state_manager import StateManager  # noqa: E402
from src.understanding.schema import Action, Facet, Intent  # noqa: E402

SESSION = "s1"


def test_unknown_session_starts_with_empty_state():
    mgr = StateManager()
    assert mgr.get_state("never-seen-before") == FilterSet()


def test_search_replaces_state_discarding_previous_filters():
    mgr = StateManager()
    mgr.apply(SESSION, Action(intent=Intent.SEARCH), FilterSet(category="milk_29", brand="juhayna", price_max=50))

    result = mgr.apply(SESSION, Action(intent=Intent.SEARCH), FilterSet(category="detergents_19"))

    assert result == FilterSet(category="detergents_19")


def test_reset_clears_state_even_with_nothing_new_resolved():
    mgr = StateManager()
    mgr.apply(SESSION, Action(intent=Intent.SEARCH), FilterSet(category="milk_29", brand="juhayna", price_max=50))

    result = mgr.apply(SESSION, Action(intent=Intent.RESET), FilterSet())

    assert result == FilterSet()


def test_reset_with_a_newly_resolved_value_is_not_forced_empty():
    """"start over, show me chocolate" is still a reset (discard prior
    state) but isn't required to end up empty -- it starts fresh with
    just this turn's own resolved values, same rule as search."""
    mgr = StateManager()
    mgr.apply(SESSION, Action(intent=Intent.SEARCH), FilterSet(category="milk_29", brand="juhayna"))

    result = mgr.apply(SESSION, Action(intent=Intent.RESET), FilterSet(category="chocolate_2"))

    assert result == FilterSet(category="chocolate_2")


@pytest.mark.parametrize("intent", [Intent.ADD_FILTER, Intent.MODIFY_FILTER])
@pytest.mark.parametrize("field,first_value,second_value", [
    ("category", "milk_29", "chocolate_2"),
    ("brand", "juhayna", "galaxy"),
    ("price_min", 20.0, 50.0),
    ("price_max", 100.0, 30.0),
])
def test_add_and_modify_filter_overwrite_one_field_leaving_others_untouched(
    intent, field, first_value, second_value
):
    mgr = StateManager()
    mgr.apply(SESSION, Action(intent=Intent.SEARCH), FilterSet(category="milk_29", brand="juhayna", price_max=200))

    resolved = FilterSet(**{field: second_value})
    result = mgr.apply(SESSION, Action(intent=intent), resolved)

    assert getattr(result, field) == second_value

    # price_min/price_max are NOT independent of each other under
    # MODIFY_FILTER the way category/brand are: the baseline here only has
    # price_max active, so a MODIFY_FILTER that states price_min alone is a
    # one-sided direction flip (see StateManager.apply's price-facet rule
    # and tests/state_manager's dedicated direction-flip tests) and
    # intentionally drops the stale price_max, rather than leaving it
    # untouched like a genuinely independent field.
    price_direction_flip = intent == Intent.MODIFY_FILTER and field == "price_min"

    for other_field in ("category", "brand", "price_min", "price_max"):
        if other_field == field:
            continue
        if price_direction_flip and other_field == "price_max":
            assert result.price_max is None
            continue
        baseline = FilterSet(category="milk_29", brand="juhayna", price_max=200)
        assert getattr(result, other_field) == getattr(baseline, other_field)


def test_add_filter_on_an_empty_session_just_sets_the_field():
    mgr = StateManager()
    result = mgr.apply(SESSION, Action(intent=Intent.ADD_FILTER), FilterSet(brand="nike"))
    assert result == FilterSet(brand="nike")


# --- MODIFY_FILTER price-bound conflict (independent audit, 2026-09): a
# turn that explicitly modifies only ONE price bound past the OTHER,
# still-retained bound used to construct an invalid FilterSet and let
# InvalidFilterError escape apply() uncaught -- a real crash on an
# ordinary utterance ("actually, minimum 100" when max=50 was already
# active). The stale opposing bound is now cleared instead. ---

def test_modify_filter_raising_min_past_retained_max_clears_the_stale_max():
    mgr = StateManager()
    mgr.apply(SESSION, Action(intent=Intent.SEARCH), FilterSet(price_min=20, price_max=50))

    result = mgr.apply(SESSION, Action(intent=Intent.MODIFY_FILTER), FilterSet(price_min=100))

    assert result.price_min == 100
    assert result.price_max is None


def test_modify_filter_lowering_max_past_retained_min_clears_the_stale_min():
    mgr = StateManager()
    mgr.apply(SESSION, Action(intent=Intent.SEARCH), FilterSet(price_min=20, price_max=50))

    result = mgr.apply(SESSION, Action(intent=Intent.MODIFY_FILTER), FilterSet(price_max=10))

    assert result.price_min is None
    assert result.price_max == 10


def test_modify_filter_non_conflicting_single_bound_preserves_the_other_bound():
    mgr = StateManager()
    mgr.apply(SESSION, Action(intent=Intent.SEARCH), FilterSet(price_min=20, price_max=50))

    result = mgr.apply(SESSION, Action(intent=Intent.MODIFY_FILTER), FilterSet(price_max=100))

    assert result.price_min == 20  # unaffected -- 20 <= 100, no conflict
    assert result.price_max == 100


def test_modify_filter_opposite_bound_landing_exactly_on_retained_bound_clears_it():
    # Live regression: "deodorant under 200" then bare "above 200" -- the
    # retained price_max (200) is not numerically GREATER than the new
    # price_min (200), so a strict `>` contradiction check let it survive,
    # producing an impossible price_min=200/price_max=200 range that
    # matches zero real products.
    mgr = StateManager()
    mgr.apply(SESSION, Action(intent=Intent.SEARCH), FilterSet(category="deodorant_1", price_max=200.0))

    result = mgr.apply(SESSION, Action(intent=Intent.MODIFY_FILTER), FilterSet(price_min=200.0))

    assert result.price_min == 200.0
    assert result.price_max is None
    assert result.category == "deodorant_1"  # active product/category context preserved


def test_modify_filter_flips_from_upper_bound_to_lower_bound():
    mgr = StateManager()
    mgr.apply(SESSION, Action(intent=Intent.SEARCH), FilterSet(price_min=300.0))

    result = mgr.apply(SESSION, Action(intent=Intent.MODIFY_FILTER), FilterSet(price_max=120.0))

    assert result.price_min is None
    assert result.price_max == 120.0


def test_modify_filter_same_direction_price_change_has_no_stale_opposite_bound():
    mgr = StateManager()
    mgr.apply(SESSION, Action(intent=Intent.SEARCH), FilterSet(price_max=90.0))

    result = mgr.apply(SESSION, Action(intent=Intent.MODIFY_FILTER), FilterSet(price_max=60.0))

    assert result.price_min is None
    assert result.price_max == 60.0


def test_modify_filter_same_direction_min_change_has_no_stale_opposite_bound():
    mgr = StateManager()
    mgr.apply(SESSION, Action(intent=Intent.SEARCH), FilterSet(price_min=40.0))

    result = mgr.apply(SESSION, Action(intent=Intent.MODIFY_FILTER), FilterSet(price_min=75.0))

    assert result.price_min == 75.0
    assert result.price_max is None


def test_modify_filter_two_sided_range_stays_intact():
    mgr = StateManager()
    result = mgr.apply(SESSION, Action(intent=Intent.SEARCH), FilterSet(price_min=100.0, price_max=200.0))

    assert result.price_min == 100.0
    assert result.price_max == 200.0


def test_modify_filter_from_active_range_to_upper_bound_drops_old_max():
    mgr = StateManager()
    mgr.apply(SESSION, Action(intent=Intent.SEARCH), FilterSet(price_min=100.0, price_max=200.0))

    result = mgr.apply(SESSION, Action(intent=Intent.MODIFY_FILTER), FilterSet(price_min=300.0))

    assert result.price_min == 300.0
    assert result.price_max is None


def test_modify_filter_from_active_range_to_lower_bound_drops_old_min():
    mgr = StateManager()
    mgr.apply(SESSION, Action(intent=Intent.SEARCH), FilterSet(price_min=100.0, price_max=200.0))

    result = mgr.apply(SESSION, Action(intent=Intent.MODIFY_FILTER), FilterSet(price_max=80.0))

    assert result.price_min is None
    assert result.price_max == 80.0


# --- direction-changing MODIFY_FILTER on a one-sided facet: intent + shape
# decide the outcome, not whether the new number happens to numerically
# clash with the old one (follow-up generalization). ---

def test_modify_filter_direction_flip_with_non_contradicting_number_replaces_facet():
    # "deodorant under 200" -> "above 150": 150 does NOT numerically
    # contradict the retained 200 (150 < 200), so a contradiction-only rule
    # would wrongly keep both and produce 150-200. MODIFY_FILTER changing
    # direction on a one-sided facet must still replace it outright.
    mgr = StateManager()
    mgr.apply(SESSION, Action(intent=Intent.SEARCH), FilterSet(category="deodorant_1", price_max=200.0))

    result = mgr.apply(SESSION, Action(intent=Intent.MODIFY_FILTER), FilterSet(price_min=150.0))

    assert result.price_min == 150.0
    assert result.price_max is None
    assert result.category == "deodorant_1"


def test_modify_filter_direction_flip_with_contradicting_number_still_replaces_facet():
    # "under 200" -> "above 250": here 250 DOES numerically contradict 200
    # too, so this must keep passing under the more general shape-based rule.
    mgr = StateManager()
    mgr.apply(SESSION, Action(intent=Intent.SEARCH), FilterSet(price_max=200.0))

    result = mgr.apply(SESSION, Action(intent=Intent.MODIFY_FILTER), FilterSet(price_min=250.0))

    assert result.price_min == 250.0
    assert result.price_max is None


def test_modify_filter_direction_flip_upper_to_lower_with_non_contradicting_number():
    # "above 200" -> "under 300": 300 does not contradict 200 either.
    mgr = StateManager()
    mgr.apply(SESSION, Action(intent=Intent.SEARCH), FilterSet(price_min=200.0))

    result = mgr.apply(SESSION, Action(intent=Intent.MODIFY_FILTER), FilterSet(price_max=300.0))

    assert result.price_min is None
    assert result.price_max == 300.0


def test_modify_filter_direction_flip_upper_to_lower_with_contradicting_number():
    # "above 200" -> "under 100": still correct under the general rule.
    mgr = StateManager()
    mgr.apply(SESSION, Action(intent=Intent.SEARCH), FilterSet(price_min=200.0))

    result = mgr.apply(SESSION, Action(intent=Intent.MODIFY_FILTER), FilterSet(price_max=100.0))

    assert result.price_min is None
    assert result.price_max == 100.0


def test_modify_filter_explicit_range_compatible_max_change_keeps_min():
    # "100-200" -> modify max to 250: an explicit two-sided range already
    # existed, and 100 is still compatible with the new max, so the
    # untouched side (min=100) is preserved rather than dropped.
    mgr = StateManager()
    mgr.apply(SESSION, Action(intent=Intent.SEARCH), FilterSet(price_min=100.0, price_max=200.0))

    result = mgr.apply(SESSION, Action(intent=Intent.MODIFY_FILTER), FilterSet(price_max=250.0))

    assert result.price_min == 100.0
    assert result.price_max == 250.0


def test_modify_filter_explicit_range_compatible_min_change_keeps_max():
    # "100-200" -> modify min to 150: 150 is still compatible with the
    # existing max, so the untouched side (max=200) is preserved.
    mgr = StateManager()
    mgr.apply(SESSION, Action(intent=Intent.SEARCH), FilterSet(price_min=100.0, price_max=200.0))

    result = mgr.apply(SESSION, Action(intent=Intent.MODIFY_FILTER), FilterSet(price_min=150.0))

    assert result.price_min == 150.0
    assert result.price_max == 200.0


def test_add_filter_with_the_same_conflicting_input_is_unchanged_out_of_scope():
    """The stale-bound-clearing rule is scoped to modify_filter only --
    add_filter genuinely adding a contradictory bound is a different,
    out-of-scope case (task instructions) and must still behave exactly
    as before this fix: InvalidFilterError propagates uncaught."""
    mgr = StateManager()
    mgr.apply(SESSION, Action(intent=Intent.SEARCH), FilterSet(price_min=20, price_max=50))

    with pytest.raises(InvalidFilterError):
        mgr.apply(SESSION, Action(intent=Intent.ADD_FILTER), FilterSet(price_min=100))


# --- get_state()/apply() must return an independent copy, not the
# internally stored object (independent audit, 2026-09): mutating the
# returned FilterSet used to silently corrupt the manager's real state
# without ever going through apply(). ---

def test_mutating_get_state_result_does_not_affect_later_get_state_calls():
    mgr = StateManager()
    mgr.apply(SESSION, Action(intent=Intent.SEARCH), FilterSet(category="milk_29"))

    fetched = mgr.get_state(SESSION)
    fetched.category = "HACKED"

    assert mgr.get_state(SESSION).category == "milk_29"


def test_mutating_apply_result_does_not_affect_stored_state():
    mgr = StateManager()

    returned = mgr.apply(SESSION, Action(intent=Intent.SEARCH), FilterSet(category="milk_29"))
    returned.category = "HACKED"

    assert mgr.get_state(SESSION).category == "milk_29"


@pytest.mark.parametrize("facet,cleared_fields", [
    (Facet.CATEGORY, ("category",)),
    (Facet.BRAND, ("brand",)),
    (Facet.PRICE, ("price_min", "price_max")),
])
def test_remove_filter_clears_only_the_targeted_facet(facet, cleared_fields):
    mgr = StateManager()
    mgr.apply(SESSION, Action(intent=Intent.SEARCH),
              FilterSet(category="milk_29", brand="juhayna", price_min=10, price_max=50))

    result = mgr.apply(SESSION, Action(intent=Intent.REMOVE_FILTER, target_facet=facet), FilterSet())

    baseline = FilterSet(category="milk_29", brand="juhayna", price_min=10, price_max=50)
    for field in ("category", "brand", "price_min", "price_max"):
        expected = None if field in cleared_fields else getattr(baseline, field)
        assert getattr(result, field) == expected


def test_remove_filter_with_no_target_facet_is_a_safe_noop():
    """The LLM leaves target_facet null when it can't tell which filter is
    meant (schema.py) -- state must be left exactly as it was, not
    guessed at or wiped."""
    mgr = StateManager()
    mgr.apply(SESSION, Action(intent=Intent.SEARCH), FilterSet(category="milk_29", brand="juhayna", price_max=50))

    result = mgr.apply(SESSION, Action(intent=Intent.REMOVE_FILTER, target_facet=None), FilterSet())

    assert result == FilterSet(category="milk_29", brand="juhayna", price_max=50)


def test_remove_filter_on_a_facet_that_is_not_active_is_a_stated_noop():
    mgr = StateManager()
    mgr.apply(SESSION, Action(intent=Intent.SEARCH), FilterSet(category="milk_29"))

    result = mgr.apply(SESSION, Action(intent=Intent.REMOVE_FILTER, target_facet=Facet.BRAND), FilterSet())

    assert result == FilterSet(category="milk_29")


def test_sessions_are_isolated_from_each_other():
    mgr = StateManager()
    mgr.apply("session-a", Action(intent=Intent.SEARCH), FilterSet(category="milk_29"))
    mgr.apply("session-b", Action(intent=Intent.SEARCH), FilterSet(category="detergents_19"))

    assert mgr.get_state("session-a") == FilterSet(category="milk_29")
    assert mgr.get_state("session-b") == FilterSet(category="detergents_19")
