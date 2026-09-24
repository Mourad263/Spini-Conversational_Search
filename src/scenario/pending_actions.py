"""Orchestrator organization refactor: resolution of a PendingAction left by
a PRIOR turn (a recommendation offer, a proposed multi-product cart add, or
a cart-checkout scope clarification), moved verbatim out of
src/scenario/orchestrator.py's _handle_message_inner(). Each resolver here
is consulted by the CURRENT turn exactly once -- the caller has already
popped the pending action (SessionActivityStore.pop_pending_action(),
"consumed once, whatever the next turn is") before calling any of these.
Each function returns a SessionMessageResponse when it resolves the pending
action this turn, or None when the current message doesn't answer it at
all -- the caller then falls through to the normal pipeline (social fast
path, then understand()) with this SAME message, exactly as before this
refactor.

The cart-checkout resolver (A2.5c) is the one deterministic, ZERO-LLM
product-reference resolution path outside src/scenario/product_selection.py
itself -- it reuses that module's exact-title/ordinal/last/fuzzy matching
unchanged, never a second resolver.
"""

from __future__ import annotations

from src.api.schemas import SessionMessageResponse
from src.recommendation.service import get_recommendation_service
from src.response import messages
from src.scenario.cart_flow import _perform_cart_add, _to_cart_out, _unpriced
from src.scenario.checkout_flow import _checkout_from_lines, _raw_message_supports_checkout_all
from src.scenario.llm_context import resolve_pending_candidates
from src.scenario.offer_response import ACCEPT, REJECT, classify_offer_response
from src.scenario.product_selection import (
    _match_named_products, _raw_message_supports_last, _raw_message_supports_ordinal,
    _selection_failure_message, _trace_resolve, _trace_set_action,
)
from src.scenario.response_builder import _response, _to_displayed_products
from src.scenario.session_activity import DisplayedProduct, PendingAction, SessionActivity, SessionActivityStore
from src.scenario.trace import RequestTrace
from src.state_manager.state_manager import StateManager
from src.understanding.schema import Action, Intent


def resolve_pending_recommendation_offer(
    session_id: str, message: str, pending: PendingAction,
    state_manager: StateManager, activity_store: SessionActivityStore, trace: RequestTrace, limit: int,
) -> SessionMessageResponse | None:
    """A2.5: pending recommendation-offer resolution (Sprint 6). Consumed
    EXACTLY once, by the very next turn after an offer, whatever that turn
    turns out to be -- the caller already popped it unconditionally BEFORE
    calling this, which is what guarantees a stale offer can never hijack a
    LATER, unrelated turn (e.g. "thank you" two turns after an unanswered
    offer): even when THIS message doesn't resolve the offer (returns
    None), the offer is already gone."""
    decision = classify_offer_response(message)
    if decision == ACCEPT:
        # Zero LLM calls, zero new search: the offering search's own
        # product ids were remembered exactly so accepting never needs a
        # second SearchAdapter call -- the EXISTING RecommendationService,
        # called once, exactly as the normal recommendation_request path
        # already does.
        activity_store.increment_turn_count(session_id)
        trace.route = "recommendation_accept"
        current_state = state_manager.get_state(session_id)
        trace.recommendation_calls += 1
        recommendations = get_recommendation_service().recommend(
            current_state.query_text, pending.shown_product_ids, current_state, limit=limit,
        )
        # Sprint 6 product-selection task: these recommendations are now
        # the latest displayed context (same rule as a normal search).
        if recommendations:
            activity_store.set_last_shown_products(session_id, _to_displayed_products(recommendations))
        return _response(
            session_id, None, "no_op", current_state,
            user_message=message, recommendations=recommendations,
            recommendation_request=True,
        )
    if decision == REJECT:
        activity_store.increment_turn_count(session_id)
        trace.route = "recommendation_declined"
        state = state_manager.get_state(session_id)
        return _response(
            session_id, None, "recommendation_declined", state,
            user_message=message,
            assistant_message_override=messages.recommendation_offer_declined_response(message),
        )
    # decision is None ("yes, show me cheese", or anything else that
    # doesn't EXACTLY match the tiny bounded accept/reject sets) -- the
    # offer above is already cleared; fall through with this SAME message
    # to the normal pipeline.
    return None


def resolve_pending_multi_add_to_cart(
    session_id: str, message: str, activity: SessionActivity, pending: PendingAction,
    state_manager: StateManager, activity_store: SessionActivityStore, trace: RequestTrace,
) -> SessionMessageResponse | None:
    """A2.5b: pending multi-product add-to-cart confirmation (cart/direct-
    checkout architecture task) -- reuses the EXACT SAME accept/reject
    classifier and "consumed once, whatever the next turn is" contract as
    the recommendation offer above, rather than inventing a second
    pending-state mechanism. Set only when a direct checkout named 2+
    products that weren't already all in the cart: the cart is the only
    durable authority for buying multiple products together, so those ids
    become real cart lines ONLY on an explicit accept here -- never
    automatically."""
    decision = classify_offer_response(message)
    if decision == ACCEPT:
        activity_store.increment_turn_count(session_id)
        trace.route = "cart_add"
        state = state_manager.get_state(session_id)
        # Re-resolved from the SAME bounded recent-shown pool the original
        # proposal drew from -- shown_product_ids are real ids already,
        # never re-validated against the LLM, just looked back up for
        # their cached name/price snapshot.
        pool = {p.id: p for p in activity_store.recent_shown_products(session_id)}
        products = [pool[pid] for pid in pending.shown_product_ids if pid in pool]
        if len(products) != len(pending.shown_product_ids):
            # Stale by the time it was accepted (the bounded window rolled
            # past it) -- never add a partial/guessed set.
            return _response(
                session_id, None, "cart_add", state, user_message=message,
                assistant_message_override=messages.selection_not_found_response(message),
            )
        # Correction: the SAME purchase-authority check as a single/multi
        # cart add elsewhere -- an unpriced product must never enter the
        # cart, and ANY unpriced item blocks the WHOLE accept (all-or-
        # nothing), never a partial add.
        unpriced = _unpriced(products)
        if unpriced:
            return _response(
                session_id, None, "cart_add", state, user_message=message,
                assistant_message_override=messages.cart_add_price_unavailable_response(
                    message, [(p.name_en, p.name_ar) for p in unpriced],
                ),
            )
        had_checkout = activity.checkout_active  # captured BEFORE the
        # mutation below (which also invalidates any open draft) so the
        # response flag reflects a REAL invalidation, never a blanket true.
        activity_store.add_to_cart(session_id, products)
        updated_cart = activity_store.get(session_id).cart
        by_id = {line.product.id: line for line in updated_cart}
        added = [(p.name_en, p.name_ar, by_id[p.id].quantity) for p in products]
        return _response(
            session_id, None, "cart_add", state, user_message=message,
            assistant_message_override=messages.cart_add_confirmation_response(message, added),
            cart=_to_cart_out(updated_cart), checkout_invalidated=had_checkout,
        )
    if decision == REJECT:
        activity_store.increment_turn_count(session_id)
        trace.route = "cart_add_declined"
        state = state_manager.get_state(session_id)
        return _response(
            session_id, None, "cart_add_declined", state, user_message=message,
            assistant_message_override=messages.multi_add_to_cart_declined_response(message),
        )
    # decision is None -- the proposal above is already cleared; fall
    # through with this SAME message to the normal pipeline.
    return None


def resolve_pending_cart_add(
    session_id: str, message: str, activity: SessionActivity, pending: PendingAction,
    state_manager: StateManager, activity_store: SessionActivityStore, trace: RequestTrace,
) -> SessionMessageResponse | None:
    """A2.5d: pending cart-add product-selection clarification -- the
    deterministic, ZERO-LLM continuation of an ambiguous cart_action="add"
    (src/scenario/cart_flow.py's handle_cart_action() now records this
    pending kind instead of leaving the clarification as text-only). This
    is the real bug this task fixes: an exact-title/ordinal answer to
    "which one?" used to fall straight through to understand() with no
    memory of the question ever having been asked, and could be misread as
    a brand-new search.

    Reuses _resolve_product_selection() completely UNCHANGED, via the SAME
    synthetic-Action-from-raw-evidence pattern resolve_pending_cart_checkout()
    uses below: an ordinal/"last" marker the RAW MESSAGE itself actually
    contains becomes a synthetic Action (no selection_text -- that needs
    real language understanding, not raw evidence). That resolver's own
    internal named/exact-title check runs regardless of the synthetic
    Action's fields, so an exact English/Arabic title resolves here too,
    with no extra code. Returns None when NEITHER check finds anything
    ("not_found") -- not a real attempt to answer THIS question at all;
    the pending state is already popped by the caller, so the normal
    pipeline gets its one real understand() call next, with compact
    session memory describing exactly this pending question and its
    candidates (see src/scenario/llm_context.py) -- letting a genuine
    semantic description ("the low-fat one") and an outright topic switch
    each be told apart there, never guessed here. Any OTHER real (if
    incomplete/ambiguous) evidence re-asks the SAME clarification, keeping
    the pending state alive for another attempt -- never a silent guess and
    never a silently dropped question.

    State-hardening task (issues 1/2): resolves ONLY against `pending`'s
    own FROZEN candidate snapshot (shown_product_ids), via
    resolve_pending_candidates() -- the SAME helper llm_context.py's
    compact CANDIDATES block uses, so the deterministic resolver here and
    whatever the LLM is shown can never disagree. Never re-reads
    last_shown_products/recent_shown_products() live -- a later, unrelated
    turn that changed the shown page must not silently widen or shift
    what this pending question is actually about."""
    state = state_manager.get_state(session_id)
    candidates = resolve_pending_candidates(activity, activity_store, session_id, pending)
    raw_ordinals = [
        i for i in range(1, len(candidates) + 1)
        if _raw_message_supports_ordinal(message, i)
    ]
    synthetic_action = Action(
        intent=Intent.NO_OP,
        selection_ordinals=raw_ordinals or None,
        selection_last=_raw_message_supports_last(message),
    )
    _trace_set_action(trace, synthetic_action)  # deterministic (no LLM) -- see this function's own docstring
    products, reason = _trace_resolve(
        trace, "pending_candidates", candidates, candidates, synthetic_action, message,
    )
    if reason == "not_found":
        return None
    if reason != "ok":
        activity_store.set_pending_action(session_id, "cart_add_pending", shown_product_ids=pending.shown_product_ids)
        trace.route = "cart_add"
        return _response(
            session_id, None, "cart_add", state, user_message=message,
            assistant_message_override=_selection_failure_message(message, reason, len(candidates)),
        )
    return _perform_cart_add(session_id, message, state, products, synthetic_action, activity, activity_store, trace)


def resolve_pending_cart_checkout(
    session_id: str, message: str, activity: SessionActivity, pending: PendingAction,
    state_manager: StateManager, activity_store: SessionActivityStore, trace: RequestTrace,
) -> SessionMessageResponse | None:
    """A2.5c: pending cart-checkout scope clarification (cart/direct-
    checkout architecture task correction) -- resolved ENTIRELY
    deterministically, BEFORE understand() is ever called, same "consumed
    once, whatever the next turn is" contract as the two resolvers above.
    The application already knows everything needed (the authoritative
    cart lines, the exact question just asked) -- no LLM call adds real
    information here, only latency/cost/a fragile dependency on the model
    happening to return NO_OP. "no"/"cancel"/"مش دلوقتي" reuses the SAME
    bounded REJECT classifier as a recommendation-offer decline.
    "all"/"everything"/"كله"/"الكل" reuses the SAME checkout-all marker
    check as an explicit "checkout all". Named references are checked via
    _match_named_products() DIRECTLY (it already works off raw text, not
    the LLM); a synthetic Action built purely from raw-text evidence
    (_raw_message_supports_ordinal()/_raw_message_supports_last()) plus
    the message itself as selection_text then goes through the ONE
    existing resolver against the cart pool -- including its own exact-
    then-fuzzy selection_text fallback, so a single shorthand name
    ("Pepsi" for a cart line "Pepsi Cola - 970 Ml") resolves the SAME way
    an ordinary selection_text turn would, never a second fuzzy engine. A
    "not_found" result (from either step) means no real product/position
    evidence was found at all -- returns None so the caller falls through
    to the normal pipeline; any OTHER real (if incomplete/ambiguous)
    evidence always gets an honest clarification, never a silent guess."""
    cart_lines = activity.cart
    state = state_manager.get_state(session_id)
    if classify_offer_response(message) == REJECT:
        activity_store.increment_turn_count(session_id)
        trace.route = "checkout_cancelled"
        return _response(
            session_id, None, "checkout_cancelled", state, user_message=message,
            assistant_message_override=messages.cart_checkout_cancelled_response(message),
        )
    if not cart_lines:
        activity_store.increment_turn_count(session_id)
        trace.route = "checkout_clarification"
        return _response(
            session_id, None, "checkout_clarification", state, user_message=message,
            assistant_message_override=messages.cart_checkout_empty_response(message),
        )
    if _raw_message_supports_checkout_all(message):
        # Correction: append to history BEFORE opening checkout -- this
        # turn's own language (which may differ from the original
        # "checkout" turn's) is what checkout_submission.py's
        # is_arabic_dominant(activity.history[-1]) will read for the final
        # order-confirmation wording.
        activity_store.append_history(session_id, message, Intent.NO_OP.value)
        activity_store.increment_turn_count(session_id)
        # trace.route is set by _checkout_from_lines() itself on success
        # (correction) -- never a redundant default here.
        return _checkout_from_lines(
            session_id, message, state, cart_lines, activity_store, trace,
            cart_ids=[line.product.id for line in cart_lines],
        )

    # State-hardening task (issues 1/2): MATCHING happens only against the
    # pending action's own frozen candidate snapshot (the SAME helper/ids
    # llm_context.py's compact CANDIDATES block uses for this exact
    # question) -- never a live re-read of the cart, which could have
    # changed via the separate cart-mutation endpoint between the question
    # and this answer. The actual checkout line/quantity is still looked
    # up from the LIVE cart in _start_from_products() below -- identity
    # comes from the frozen snapshot, quantity/authority from current cart
    # truth, never the other way around.
    cart_pool = resolve_pending_candidates(activity, activity_store, session_id, pending)
    by_id = {line.product.id: line for line in cart_lines}

    def _start_from_products(resolved: list[DisplayedProduct]):
        # Defensive: a resolved candidate may have since been removed from
        # the live cart (e.g. via the separate cart-mutation endpoint) --
        # never crash, ask again instead of checking out a stale product.
        still_in_cart = [p for p in resolved if p.id in by_id]
        if len(still_in_cart) != len(resolved):
            return _clarify("not_found")
        activity_store.append_history(session_id, message, Intent.NO_OP.value)  # see above
        activity_store.increment_turn_count(session_id)
        selected_lines = [by_id[p.id] for p in resolved]
        return _checkout_from_lines(
            session_id, message, state, selected_lines, activity_store, trace,
            cart_ids=[line.product.id for line in selected_lines],
        )

    def _clarify(reason: str):
        activity_store.increment_turn_count(session_id)
        trace.route = "checkout_clarification"
        return _response(
            session_id, None, "checkout_clarification", state, user_message=message,
            assistant_message_override=_selection_failure_message(message, reason, len(cart_pool), cart=True),
        )

    named = _match_named_products(message, cart_pool)
    if named is not None:
        products, reason = named
        return _start_from_products(products) if reason == "ok" else _clarify(reason)

    raw_ordinals = [i for i in range(1, len(cart_pool) + 1) if _raw_message_supports_ordinal(message, i)]
    synthetic_action = Action(
        intent=Intent.NO_OP,
        selection_ordinals=raw_ordinals or None,
        selection_last=_raw_message_supports_last(message),
        selection_text=message.strip() or None,
    )
    _trace_set_action(trace, synthetic_action)  # deterministic (no LLM) -- see this function's own docstring
    # named is already known to be None for this exact message/pool
    # (checked directly above) -- this call's own internal named check
    # repeats that same (cheap, deterministic) lookup once more and falls
    # through it exactly as before, straight to ordinal/last, and finally
    # selection_text's own exact-then-fuzzy match against the cart pool
    # (the single-shorthand-name case).
    products, reason = _trace_resolve(trace, "cart", cart_pool, cart_pool, synthetic_action, message)
    if reason == "ok":
        return _start_from_products(products)
    if reason == "not_found":
        # No real evidence of any kind -- this wasn't a real attempt to
        # answer the cart question at all; falls through to the normal
        # pipeline with this SAME message.
        return None
    return _clarify(reason)
