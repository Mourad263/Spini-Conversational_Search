"""Sprint 3 State Manager (tasks.md T027/T033/T034): deterministic,
non-LLM per-session filter state, updated turn over turn from the
`intent` already classified by the Understanding layer
(src/understanding/llm.py) and the FilterSet already resolved by
Validation (src/validation/validate.py).

Deliberately contains no LLM judgment. Whether a turn continues the
active search or starts a new one is decided upstream -- the Understanding
LLM sees a bounded conversation-history window (llm.py's HISTORY_WINDOW)
precisely so that judgment is already baked into `action.intent` by the
time it reaches here. Applying it is then pure bookkeeping:

- search / reset -> discard prior state, keep only this turn's resolved
  values (an empty FilterSet for a bare "start over"/"reset" with nothing
  else said, or a fresh set of filters for "start over, show me
  chocolate").
- add_filter / modify_filter -> merge this turn's non-null resolved
  values into the existing state, leaving every other field untouched.
  Whether a field that changes was previously empty ("add") or already
  had a value ("modify") is otherwise a UX distinction, not a different
  merge rule -- except for one price-only case: modify_filter changing
  just one bound past the OTHER, still-retained bound clears that stale
  bound instead of constructing an invalid FilterSet (see apply()).
- remove_filter -> clear exactly the field named by
  `action.target_facet` (schema.py), leaving the rest untouched. If
  target_facet is null (the LLM couldn't tell which filter was meant),
  this is a safe no-op -- state is left unchanged rather than guessing
  what to clear.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.search_adapter.adapter import FilterSet  # noqa: E402
from src.understanding.schema import Action, Facet, Intent  # noqa: E402


class StateManager:
    """In-memory per-session FilterSet store. This project's scale (a
    handful of concurrent conversations, not a production multi-user
    deployment) doesn't need a persistence layer."""

    def __init__(self) -> None:
        self._sessions: dict[str, FilterSet] = {}

    def get_state(self, session_id: str) -> FilterSet:
        # A copy, not the stored object itself -- otherwise a caller
        # mutating the returned FilterSet in place would silently corrupt
        # the manager's real internal state without ever going through
        # apply() (confirmed: `get_state(id) is self._sessions[id]` was
        # True before this fix).
        stored = self._sessions.get(session_id)
        return copy.copy(stored) if stored is not None else FilterSet()

    def apply(self, session_id: str, action: Action, resolved: FilterSet) -> FilterSet:
        current = self.get_state(session_id)

        if action.intent in (Intent.SEARCH, Intent.RESET):
            new_state = FilterSet(
                category=resolved.category,
                brand=resolved.brand,
                price_min=resolved.price_min,
                price_max=resolved.price_max,
                query_text=resolved.query_text,
            )
        elif action.intent in (Intent.ADD_FILTER, Intent.MODIFY_FILTER):
            price_min = resolved.price_min if resolved.price_min is not None else current.price_min
            price_max = resolved.price_max if resolved.price_max is not None else current.price_max
            if action.intent == Intent.MODIFY_FILTER:
                # Scoped to modify_filter only -- add_filter genuinely
                # adding a contradictory bound is a different, out-of-scope
                # case (research.md).
                #
                # Shape-aware, not numeric-aware: whether a stale bound
                # survives depends on what kind of price facet was ACTIVE
                # before this turn, not on whether the new number happens
                # to numerically clash with it.
                #
                # If the facet was already ONE-SIDED (only a min or only a
                # max was ever stated) and this turn states the OPPOSITE
                # side alone, the user has changed direction entirely --
                # "under 200" then "above 150" means "price_min=150" full
                # stop, not an accidental 150-200 range, even though 150
                # does not numerically contradict 200. The old side is
                # always dropped here, unconditionally.
                #
                # Otherwise -- the facet was already a genuine two-sided
                # range, or this turn's new bound is on the SAME side that
                # was already active -- the previous side is only dropped
                # when keeping it would be a real numeric contradiction
                # (min >= max). A compatible adjustment to one side of an
                # explicit range ("100-200" -> modify max to 250) keeps the
                # other side untouched.
                if resolved.price_min is not None and resolved.price_max is None:
                    if current.price_min is None and current.price_max is not None:
                        price_max = None  # one-sided direction flip
                    elif price_max is not None and price_min >= price_max:
                        price_max = None  # genuine numeric contradiction
                elif resolved.price_max is not None and resolved.price_min is None:
                    if current.price_max is None and current.price_min is not None:
                        price_min = None  # one-sided direction flip
                    elif price_min is not None and price_min >= price_max:
                        price_min = None  # genuine numeric contradiction
            new_state = FilterSet(
                category=resolved.category if resolved.category is not None else current.category,
                brand=resolved.brand if resolved.brand is not None else current.brand,
                price_min=price_min,
                price_max=price_max,
                # Sprint 6: resolved.query_text can be None for TWO different
                # reasons -- nothing product-related was said this turn, OR
                # something WAS said but validate() correctly reduced it to
                # None because the grounded category already represents it
                # ("cheese instead" -> category=cheese, query_text=None).
                # Falling back to `current.query_text` in the second case
                # would resurrect a stale product concept from a PRIOR turn
                # ("chocolate milk") underneath the NEW category. The
                # Action's own raw_query_text says whether a product concept
                # was expressed this turn at all, independent of how it
                # resolved -- that's the signal to key off, not None-ness of
                # the resolved value.
                query_text=(
                    resolved.query_text if action.raw_query_text is not None
                    else current.query_text
                ),
            )
        elif action.intent == Intent.REMOVE_FILTER:
            new_state = FilterSet(
                category=current.category,
                brand=current.brand,
                price_min=current.price_min,
                price_max=current.price_max,
                query_text=current.query_text,
            )
            if action.target_facet == Facet.CATEGORY:
                new_state.category = None
            elif action.target_facet == Facet.BRAND:
                new_state.brand = None
            elif action.target_facet == Facet.PRICE:
                new_state.price_min = None
                new_state.price_max = None
            # target_facet is None -> no-op, state unchanged (see module docstring)
        else:
            raise ValueError(f"unhandled intent: {action.intent!r}")

        self._sessions[session_id] = new_state
        return copy.copy(new_state)  # same reason as get_state(): the caller's copy
        # must be independent of the internally stored one

    def clear(self, session_id: str) -> None:
        """Drop this session's stored FilterSet entirely (Sprint 6
        lifecycle task) -- used only for genuine session EXPIRATION
        (src/scenario/session_activity.py's SESSION_TTL_SECONDS), never
        for an ordinary "start over"/RESET turn, which already goes
        through apply() and keeps the session id alive. A never-seen or
        already-cleared session_id is a no-op."""
        self._sessions.pop(session_id, None)
