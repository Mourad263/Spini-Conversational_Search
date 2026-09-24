"""Sprint 4 deterministic orchestration helpers (design pass, Part 2):
narrow, evidence-driven repairs for two known LLM boundary cases, plus the
deterministic recommendation/exploration search policy. Deliberately NOT
a generic LLM-correction framework -- only the two invariants below are
implemented, both confirmed against real understand() output (Sprint 4
Part 1 verification passes).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.scenario.session_activity import SessionActivity  # noqa: E402
from src.search_adapter.adapter import FilterSet  # noqa: E402
from src.understanding.schema import Action, Intent  # noqa: E402

RELAXATION_FACTOR = 1.20
MAX_RELAXATION_STEPS = 3


def normalize_target_facet(action: Action) -> Action:
    """target_facet is only ever meaningful for remove_filter (schema.py).
    Application-level invariant, not something to rely on the
    probabilistic model to obey perfectly -- confirmed live (Sprint 4
    Part 1 verification) that gpt-5.4-mini can populate it on a
    modify_filter turn. Returns a new Action (never mutates `action`);
    every other field is untouched."""
    if action.intent == Intent.REMOVE_FILTER or action.target_facet is None:
        return action
    return action.model_copy(update={"target_facet": None})


def has_searchable_target(filters: FilterSet) -> bool:
    """A FilterSet has a meaningful product target only when at least one
    of query_text/category/brand is present -- price alone is not a
    target: recommending against a price-only state would mean browsing
    the entire catalog, not exploring a real product context (Sprint 4
    design pass, section 6)."""
    return filters.query_text is not None or filters.category is not None or filters.brand is not None


def normalize_recommendation_intent(action: Action, current_state: FilterSet) -> Action:
    """ONE Action-normalization function for a single general concept: when
    recommendation_request=True, the EFFECTIVE intent must reflect whether
    the turn actually states a new concrete value, never the LLM's raw
    intent label alone. Unifies two previously separate, mutually-
    exclusive repairs (a real intent and NO_OP can never both be
    action.intent at once, so exactly one branch below ever applies) --
    both confirmed live, real gpt-5.4-mini boundary cases:

    Direction 1 (NO_OP that should have been a real intent -- Sprint 4
    Part 1 micro-verification): a concrete PRICE bound stated on the
    CURRENT turn -- the only field confirmed to trigger this shape live --
    means the turn is not a true no-op. The effective intent is chosen
    from the CURRENT canonical StateManager state, never guessed by the
    LLM.

    Direction 2 (a state-changing intent that should have been NO_OP --
    Sprint 6 recommendation-context task): NO field at all is stated
    (raw_query_text/category_hint/brand_text/free_from/price_min/
    price_max all null) -- a pure recommendation-continuation phrase
    ("recommend me more") misclassified as a fresh SEARCH, which would
    otherwise let StateManager.apply() wipe the active canonical state
    (e.g. "chips", price>=20) for no reason.

    Every other shape -- recommendation_request=False, a NO_OP that
    states nothing, or a real intent that already states a new product/
    category/brand/price -- passes through completely unchanged; only
    these two exact, evidence-confirmed gaps are repaired, never a
    broader "any field implies a real intent" rule (that would be a
    genuine behavior change beyond what either confirmed case needs)."""
    if not action.recommendation_request:
        return action

    if action.intent == Intent.NO_OP:
        if action.price_min is None and action.price_max is None:
            return action
        if not has_searchable_target(current_state):
            effective_intent = Intent.SEARCH
        else:
            bound_already_set = (
                (action.price_min is not None and current_state.price_min is not None)
                or (action.price_max is not None and current_state.price_max is not None)
            )
            # MODIFY_FILTER when ANY corresponding bound already exists (this also
            # correctly engages StateManager.apply()'s stale-opposite-bound-clearing
            # safety net, which is scoped to modify_filter only); plain ADD_FILTER only
            # when neither newly-stated bound was already present.
            effective_intent = Intent.MODIFY_FILTER if bound_already_set else Intent.ADD_FILTER
        return action.model_copy(update={"intent": effective_intent})

    if action.intent in (Intent.SEARCH, Intent.ADD_FILTER, Intent.MODIFY_FILTER):
        states_new_value = any([
            action.raw_query_text, action.category_hint, action.brand_text, action.free_from,
            action.price_min is not None, action.price_max is not None,
        ])
        if states_new_value:
            return action
        return action.model_copy(update={"intent": Intent.NO_OP})

    return action


@dataclass
class RecommendationOutcome:
    needs_clarification: bool  # True: no meaningful product target exists at all --
    # Part 3 must ask for one, never search the whole catalog.
    effective_filters: FilterSet | None  # the TEMPORARY FilterSet for Part 3 to hand to
    # SearchAdapter -- None when needs_clarification or relaxation_exhausted. Always a
    # fresh FilterSet, never the canonical object itself (see module docstring).
    relaxation_step: int  # activity.relaxation_step AFTER this call
    relaxation_applied: bool  # True only when THIS call actually produced a new price band
    relaxation_exhausted: bool  # True when MAX_RELAXATION_STEPS was already reached
    # before this call -- Part 3 should ask for a new explicit budget/filter instead.


def derive_recommendation_search(current_state: FilterSet, activity: SessionActivity) -> RecommendationOutcome:
    """Sprint 4 deterministic recommendation/exploration policy (design
    pass, sections 7-10). The LLM never chooses a product or a budget --
    this reads the CURRENT canonical StateManager state and, when a price
    bound is active, derives a bounded, deterministically-widened
    TEMPORARY effective FilterSet. StateManager.apply() is never called
    here and the canonical FilterSet is never mutated -- callers pass the
    result of StateManager.get_state() in as `current_state`, and this
    function only ever reads it, constructing fresh FilterSet objects for
    any output."""
    if not has_searchable_target(current_state):
        return RecommendationOutcome(
            needs_clarification=True,
            effective_filters=None,
            relaxation_step=activity.relaxation_step,
            relaxation_applied=False,
            relaxation_exhausted=False,
        )

    if current_state.price_max is None:
        # Use canonical state as-is -- never invent a price ceiling that was never
        # stated. relaxation_step is left untouched: there is nothing to relax.
        return RecommendationOutcome(
            needs_clarification=False,
            effective_filters=FilterSet(
                category=current_state.category,
                brand=current_state.brand,
                price_min=current_state.price_min,
                price_max=None,
                query_text=current_state.query_text,
            ),
            relaxation_step=activity.relaxation_step,
            relaxation_applied=False,
            relaxation_exhausted=False,
        )

    if activity.relaxation_step >= MAX_RELAXATION_STEPS:
        return RecommendationOutcome(
            needs_clarification=False,
            effective_filters=None,
            relaxation_step=activity.relaxation_step,
            relaxation_applied=False,
            relaxation_exhausted=True,
        )

    # Explore the NEXT price BAND above the previously-shown ceiling, not a wider
    # 0..relaxed_max range -- re-querying from the original price_min (or the
    # canonical price_max itself, as a floor) would just re-return the same
    # already-seen products (Sprint 4 Part 2 micro-correction, confirmed against
    # real product feedback). Both bounds are computed fresh from the canonical
    # price_max each call -- band N is [max*1.20^(N-1), max*1.20^N] -- so the
    # user's own canonical price_min plays no part in the band's lower edge; it is
    # deliberately superseded for this temporary, system-generated exploration
    # range only (the canonical FilterSet itself is never touched either way).
    next_step = activity.relaxation_step + 1
    band_min = round(current_state.price_max * (RELAXATION_FACTOR ** (next_step - 1)), 2)
    band_max = round(current_state.price_max * (RELAXATION_FACTOR ** next_step), 2)
    activity.relaxation_step = next_step

    return RecommendationOutcome(
        needs_clarification=False,
        effective_filters=FilterSet(
            category=current_state.category,
            brand=current_state.brand,
            price_min=band_min,
            price_max=band_max,
            query_text=current_state.query_text,
        ),
        relaxation_step=next_step,
        relaxation_applied=True,
        relaxation_exhausted=False,
    )
