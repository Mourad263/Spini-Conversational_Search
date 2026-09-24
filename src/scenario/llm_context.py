"""Deterministic, compact session-memory context for the LLM understanding
layer (src/understanding/llm.py) -- built entirely from data the backend
already has (SessionActivity, StateManager's FilterSet, the current
PendingAction), never a second LLM call summarizing anything. The LLM
reads this to interpret language; it never returns state, never owns a
transition -- see src/scenario/orchestrator.py's handle_message(), the
only caller, which wires this in via understanding.llm.
set_session_context() (the SAME per-call contextvar convention
set_selectable_products_shown() already uses, so every existing stubbed
understand_fn in tests keeps working unchanged).

Conversation state and compact LLM memory:

    STATE IS DETERMINISTIC.
    SESSION MEMORY IS CONTEXTUAL.
    THE LLM MAY READ SESSION MEMORY TO UNDERSTAND LANGUAGE.
    THE LLM MUST NEVER OWN, SET, OR TRANSITION THE STATE.

Two layers, plus a conditional third:

Layer 1 (CURRENT_CONTEXT) -- a tiny read-only snapshot of the authoritative
state relevant to language understanding: the derived conversation phase,
cart/checkout/shown-product counts, the active search subject. The LLM
never returns a modified version of this -- it only reads it.

Layer 2 (RECENT_CONTEXT) -- a bounded window of recent turns, each as the
user's own raw text plus a SHORT, deterministic outcome summary (never an
LLM-generated summary, never the full assistant reply or product cards --
see summarize_turn_outcome()).

CANDIDATES (conditional) -- only when a pending selection is open: the
bounded, RELEVANT candidate product/cart-line names the current question
is actually about -- never the wider catalog, never an unrelated older
search's results.
"""

from __future__ import annotations

from src.api.schemas import SessionMessageResponse
from src.scenario.session_activity import DisplayedProduct, PendingAction, SessionActivity, SessionActivityStore
from src.scenario.trace import RequestTrace
from src.search_adapter.adapter import FilterSet

RECENT_TURNS_SHOWN = 3  # bounded, smaller than HISTORY_WINDOW's raw-text
# turns (src/understanding/llm.py) -- an outcome summary is far more compact
# than raw text, so this doesn't need to match that exact number, just the
# SAME "small bounded window" philosophy (Part 10 token discipline).

# Conversation phase for each pending-action kind -- DERIVED display names
# only; `PendingAction.kind` (session_activity.py) remains the one
# authoritative string actually stored/compared against in routing code.
_PHASE_BY_PENDING_KIND = {
    "cart_add_pending": "PENDING_CART_ADD_SELECTION",
    "cart_checkout_pending": "PENDING_CART_CHECKOUT_SELECTION",
    "checkout_pending_selection": "PENDING_CHECKOUT_SELECTION",
    "multi_add_to_cart_pending": "PENDING_MULTI_ADD_CONFIRMATION",
    "recommendation_offer": "PENDING_RECOMMENDATION_CONFIRMATION",
}


def derive_conversation_phase(activity: SessionActivity) -> str:
    """The current conversational phase, DERIVED from already-authoritative
    session fields -- never an independently mutable variable, so this can
    never become a second source of truth for what pending_action/
    checkout_active/cart/last_shown_products already represent. Priority:
    an open pending question always wins (it demands the very next
    answer), then an open checkout draft, then ordinary browsing (something
    has been shown or is in the cart), else a fresh/idle session."""
    if activity.pending_action is not None:
        return _PHASE_BY_PENDING_KIND.get(activity.pending_action.kind, "PENDING")
    if activity.checkout_active:
        return "CHECKOUT_ACTIVE"
    if activity.last_shown_products or activity.cart:
        return "BROWSING"
    return "IDLE"


# route -> a short, fixed, deterministic phrase, built only from data
# src/scenario/trace.py's RequestTrace already records elsewhere in this
# package -- never LLM free-text, never the customer-facing
# assistant_message. "cart_add" and "checkout_clarification" cover more
# than one real outcome (a successful mutation vs. one more clarifying
# question), so those two are special-cased in summarize_turn_outcome()
# below using trace.pending_action_after rather than route alone.
_ROUTE_OUTCOME = {
    "blocked": "message was blocked (repeated too quickly)",
    "social_fast_path": "exchanged a social greeting",
    "error": "understanding failed; asked the user to try again",
    "memory_recall": "answered a question about the conversation itself",
    "product_selection": "acknowledged a shown-product reference",
    "multi_direct_requires_cart": "proposed adding missing items to the cart before checkout",
    "off_topic": "message was off-topic; refocused on shopping",
    "reset": "conversation was reset/started over",
    "out_of_catalog": "requested concept is not in this catalog",
    "cart_view": "showed the cart contents",
    "cart_clear": "cleared the cart",
    "cart_remove": "removed item(s) from the cart",
    "cart_update_quantity": "updated a cart item's quantity",
    "recommendation_accept": "accepted the recommendation offer",
    "recommendation_declined": "declined the recommendation offer",
    "cart_add_declined": "declined adding the proposed items to the cart",
    "checkout_cancelled": "cancelled the checkout",
    "checkout_price_unavailable": "checkout blocked: price unavailable",
    "checkout_start": "started a checkout draft",
}


def summarize_turn_outcome(trace: RequestTrace, response: SessionMessageResponse) -> str:
    """A short, human-readable, fully DETERMINISTIC summary of what this
    turn actually did -- built only from already-computed route/trace/
    response data, never a second LLM call asked to summarize itself (see
    this module's docstring). Used both for the terminal trace and as
    Layer 2 conversational memory for a LATER turn's compact LLM context."""
    route = trace.route
    if route == "shopping_search":
        subject = response.resolved_category or response.resolved_brand or ""
        if response.zero_result:
            return f"searched for {subject} -- no results" if subject else "search returned no results"
        text = f"displayed {len(response.products)} {subject} products".replace("  ", " ")
        return text.strip()
    if route == "recommendation":
        if response.recommendations:
            return f"showed {len(response.recommendations)} recommended products"
        return "explored recommendations (nothing new to show)"
    if route == "cart_add":
        if trace.pending_action_after == "cart_add_pending":
            return "cart add was ambiguous; asked user to identify one shown product"
        return "added item(s) to the cart"
    if route == "checkout_clarification":
        if trace.pending_action_after == "cart_checkout_pending":
            return "asked whether to checkout all cart items or specific ones"
        if trace.pending_action_after == "checkout_pending_selection":
            return "checkout selection was ambiguous; asked user to identify the product"
        return "asked for more information before checkout"
    if route in _ROUTE_OUTCOME:
        return _ROUTE_OUTCOME[route]
    return f"turn resolved (route={route})"


def _candidate_lines(candidates: list[DisplayedProduct]) -> list[str]:
    lines = []
    for i, p in enumerate(candidates, start=1):
        name = p.name_en or ""
        if p.name_ar:
            name = f"{name} | {p.name_ar}" if name else p.name_ar
        lines.append(f"{i}. {name}")
    return lines


def _format_candidates(candidates: list[DisplayedProduct], activity: SessionActivity, kind: str) -> list[str]:
    """cart_checkout_pending candidates also show the real cart quantity
    (looked up from the LIVE cart -- quantity itself isn't part of a
    product's identity snapshot); every other kind just shows the name."""
    if kind == "cart_checkout_pending":
        qty_by_id = {line.product.id: line.quantity for line in activity.cart}
        lines = []
        for i, p in enumerate(candidates, start=1):
            name = p.name_en or p.name_ar or ""
            lines.append(f"{i}. {name} (x{qty_by_id.get(p.id, 1)})")
        return lines
    return _candidate_lines(candidates)


def resolve_pending_candidates(
    activity: SessionActivity, activity_store: SessionActivityStore, session_id: str, pending: PendingAction,
) -> list[DisplayedProduct]:
    """State-hardening task: THE one place a PendingAction's frozen
    candidate ids (`shown_product_ids`) resolve back to real
    DisplayedProduct records -- used identically by the deterministic
    pending resolvers (src/scenario/pending_actions.py) and the compact
    LLM CANDIDATES block (build_llm_session_context() below), so the two
    can never disagree about what is actually being disambiguated.

    Looks ids up against the union of the bounded recent-shown window and
    the cart -- covers every pending kind's real candidate source (shown-
    product-sourced kinds and cart-sourced kinds alike) in one lookup.
    Never re-derives a fresh "current" pool: the candidates are a FROZEN
    snapshot taken at the moment the pending question was asked, and this
    only looks each already-decided id back up for its (still-current)
    display/matching data -- preserving `pending.shown_product_ids`'s own
    order."""
    pool = {p.id: p for p in activity_store.recent_shown_products(session_id)}
    for line in activity.cart:
        pool.setdefault(line.product.id, line.product)
    return [pool[pid] for pid in pending.shown_product_ids if pid in pool]


def build_llm_session_context(
    activity: SessionActivity, state: FilterSet, pending: PendingAction | None,
    activity_store: SessionActivityStore, session_id: str,
) -> str | None:
    """The ONE helper responsible for building the compact, read-only
    session context understand() may read. Returns None for a session
    with nothing useful to add yet (a brand new session: no pending
    action, no prior turns) -- the LLM prompt then stays exactly as small
    as it already was for that common case (Part 10 token discipline).

    `pending` is the ALREADY-POPPED PendingAction a caller may still be
    holding this turn (src/scenario/orchestrator.py's A2.5 block pops it
    before any resolver runs) -- passed explicitly rather than re-read
    from `activity` because by the time this is called,
    activity.pending_action is already None for this turn; the LLM still
    needs to know what question is actually open."""
    phase = _PHASE_BY_PENDING_KIND.get(pending.kind, "PENDING") if pending is not None else derive_conversation_phase(activity)
    if phase == "IDLE" and not activity.turns:
        return None

    active_search = state.query_text or state.category or state.brand or "none"
    lines = [
        "CURRENT_CONTEXT",
        f"phase: {phase}",
        f"checkout_active: {'true' if activity.checkout_active else 'false'}",
        f"cart_item_count: {len(activity.cart)}",
        f"active_search: {active_search}",
        f"shown_products_count: {len(activity.last_shown_products)}",
        f"pending_action: {pending.kind if pending is not None else 'none'}",
    ]

    recent_turns = activity.turns[-RECENT_TURNS_SHOWN:]
    if recent_turns:
        lines += ["", "RECENT_CONTEXT"]
        for turn in recent_turns:
            lines.append(f"user: {turn.message}")
            lines.append(f"outcome: {turn.outcome or f'turn resolved (intent={turn.intent})'}")

    # State-hardening task (issue 2): candidates ALWAYS come from the
    # pending action's own frozen snapshot (pending.shown_product_ids),
    # via the SAME resolve_pending_candidates() helper the deterministic
    # resolvers use -- never re-derived from last_shown_products/cart by
    # kind here, so the LLM can never be shown a different candidate set
    # than what the resolver will actually match against.
    if pending is not None and pending.shown_product_ids:
        candidates = resolve_pending_candidates(activity, activity_store, session_id, pending)
        candidate_lines = _format_candidates(candidates, activity, pending.kind)
        if candidate_lines:
            header = (
                "CANDIDATES (cart items -- identify which one(s), if any)"
                if pending.kind == "cart_checkout_pending"
                else "CANDIDATES (identify which one, if any, this message refers to)"
            )
            lines += ["", header, *candidate_lines]

    return "\n".join(lines)
