"""Orchestrator organization refactor: cart-specific deterministic helpers,
moved verbatim out of src/scenario/orchestrator.py -- the purchase-
authority check every cart/checkout mutation reuses (_unpriced), quantity
validation, the cart's own API-response snapshot, the combined cart+recent-
shown NAMED-resolution pool, the raw-message marker that tells a cart-scoped
reference ("checkout Pepsi from my cart") apart from an ordinary one, and the
structured test for whether a cart add actually named a product at all. These are pure
functions (or read-only pool builders); the actual cart mutations
(add_to_cart/remove_from_cart/set_cart_quantity/clear_cart) remain the
SAME SessionActivityStore methods orchestrator.py already calls -- this
module never re-implements them.
"""

from __future__ import annotations

import re

from rapidfuzz import fuzz

from src.api.schemas import CartLineOut, CartOut, SessionMessageResponse
from src.checkout.demo_order import DemoOrderValidationError, validate_quantity
from src.ingestion.text_normalize import normalize_arabic, normalize_latin
from src.pricing import effective_price
from src.response import messages
from src.scenario.llm_context import resolve_pending_candidates
from src.scenario.product_selection import (
    _ARABIC_INDIC_DIGITS, _exact_title_matches, _normalize_field, _selection_failure_message, _trace_resolve,
)
from src.scenario.response_builder import _response
from src.scenario.session_activity import (
    CartLine, DisplayedProduct, PendingAction, SessionActivity, SessionActivityStore,
)
from src.scenario.trace import RequestTrace
from src.search_adapter.adapter import FilterSet
from src.understanding.schema import Action, Intent


def _unpriced(products: list[DisplayedProduct]) -> list[DisplayedProduct]:
    """Cart/direct-checkout architecture task (correction): products whose
    src/pricing.py effective_price() is unknown -- the ONE purchase-
    authority check, reused (never re-implemented) everywhere a product is
    about to enter the cart OR a checkout draft. An unpriced product may
    never enter the cart at all (not just never reach checkout) -- see
    add_to_cart() call sites below."""
    return [p for p in products if effective_price(p.price, p.special_price) is None]


def _validated_cart_quantity(quantity: int | None, *, default: int | None) -> int | None:
    """Cart/direct-checkout architecture task (correction): the SAME
    authoritative rule checkout submission itself enforces (src/checkout/
    demo_order.py's validate_quantity()), reused here rather than a second
    ad hoc ">=1" check. `default` is returned when `quantity` is None
    (nothing stated) -- pass None to make an explicit quantity mandatory
    (update_quantity); pass a real int (1) to make a missing quantity
    imply that default (add). Returns None when a STATED quantity fails
    validation -- the caller must clarify, never silently clamp/guess."""
    if quantity is None:
        return default
    try:
        return validate_quantity(quantity)
    except DemoOrderValidationError:
        return None


def _cart_and_recent_pool(activity: SessionActivity, activity_store: SessionActivityStore, session_id: str) -> list[DisplayedProduct]:
    """Cart/direct-checkout architecture task (correction): the bounded
    NAMED-resolution pool for a direct checkout request that names 2+ (or
    1) products without explicitly mentioning the cart -- cart products
    (freshest snapshot, since the cart line's own price is the ONE the
    user is actually about to pay) plus the bounded recent-shown window,
    deduped by id. Reused ONLY by the named-matching path
    (_resolve_product_selection()'s `recent_shown` parameter) -- never by
    positional/vague references, which stay scoped to `current_shown`
    (last_shown_products) exactly as before; this function does not touch
    that parameter at all, so it cannot leak cart access into "first"/
    "third"/"last"."""
    seen: dict[int, DisplayedProduct] = {}
    for line in activity.cart:
        seen[line.product.id] = line.product
    for p in activity_store.recent_shown_products(session_id):
        seen.setdefault(p.id, p)
    return list(seen.values())


def _to_cart_out(cart: list[CartLine]) -> CartOut:
    """Cart/direct-checkout architecture task: the cart's current, server-
    authoritative snapshot for the API response -- unit_price/line_total
    come only from src/pricing.py's effective_price() against the line's
    own cached product snapshot, never a fresh catalog lookup. Always a
    real, known price (correction): an unpriced product can no longer
    enter the cart at all (see _unpriced()/add_to_cart() call sites), so
    `total` is always the REAL full cart total, never a known-price
    subset."""
    lines = []
    total = 0.0
    for line in cart:
        unit_price = effective_price(line.product.price, line.product.special_price)
        line_total = unit_price * line.quantity
        total += line_total
        lines.append(CartLineOut(
            id=line.product.id, name_en=line.product.name_en, name_ar=line.product.name_ar,
            quantity=line.quantity, unit_price=unit_price, line_total=line_total,
        ))
    return CartOut(lines=lines, total=total)


_ECHO_SIMILARITY = 85  # a selection_text this close to the WHOLE utterance is the
# utterance itself, not a description of a product inside it.


def _normalize_generic_add(action: Action, message: str) -> Action:
    """Deterministic safety net for Action.generic_reference (hardening): the
    LLM sometimes copies the whole command into selection_text ("add to
    cart" -> selection_text="add to cart") without marking the reference
    generic. A selection_text that is effectively the user's ENTIRE
    utterance carries no product identity -- a product description is a
    part of the message, never all of it -- so it is dropped and the add is
    treated as an unresolved generic reference. Purely structural
    (normalized whole-string similarity): no phrase or language list, so it
    holds for any wording. Real deterministic evidence is untouched: a
    stated position/"last" and any exact/named product in the RAW message
    are still resolved from the message itself by the resolver, so they win
    regardless of what selection_text held."""
    text = action.selection_text
    if not text or action.generic_reference is not None:
        return action
    if fuzz.ratio(_normalize_field(text), _normalize_field(message)) < _ECHO_SIMILARITY:
        return action
    return action.model_copy(update={"selection_text": None, "generic_reference": "singular"})


def _names_a_product(action: Action) -> bool:
    """Generic cart-add task: True only when the Action carries real product-
    reference evidence -- a stated position, "last", or a product
    description. An add with NONE of these ("add to cart", "add it", "حط في
    السلة", "حطهم في الكارت") is a GENERIC cart action, not an unmatched
    product name. Decided from the structured Action alone (what the user
    actually referred to), never from a list of pronouns/commands, so it
    holds for any wording or language the LLM can read.
    `generic_reference` overrides selection_text: the LLM declaring the
    reference generic means any selection_text it copied is the command
    itself, not a product (see Action.generic_reference)."""
    if action.selection_ordinals or action.selection_last:
        return True
    return bool(action.selection_text) and action.generic_reference is None


_CART_MARKERS = ("cart", "سلة")  # English "cart"; Arabic "سلة" ("the cart" --
# "السلة" -- already contains "سلة" as a substring, so one marker covers
# both). Cart/direct-checkout architecture task: the deterministic signal
# that a checkout/selection reference means the CART specifically ("checkout
# Pepsi from my cart", "remove Pepsi from cart") rather than shown-product
# authority -- never trusted from an LLM field, exactly like every other
# raw-message marker in this module.


def _raw_message_mentions_cart(raw_message: str) -> bool:
    """True only when the user's own words actually mention the cart --
    never inferred from intent alone (see _resolve_product_selection() for
    the same raw-message-authority convention)."""
    norm = normalize_latin(raw_message) + normalize_arabic(raw_message)
    return any(marker in norm for marker in _CART_MARKERS)


def _perform_cart_add(
    session_id: str, message: str, state: FilterSet, products: list[DisplayedProduct],
    action: Action, activity: SessionActivity, activity_store: SessionActivityStore, trace: RequestTrace,
) -> SessionMessageResponse:
    """Deterministic conversation-state/compact-memory task: the ONE place
    an already-resolved (unambiguous) cart-add product list actually
    becomes a cart mutation -- shared by handle_cart_action()'s own "add"
    resolution below AND src/scenario/pending_actions.py's
    resolve_pending_cart_add() (the deterministic continuation of a prior
    ambiguous add), so the purchase-authority/quantity/mutation rules are
    never duplicated between them. `products` must already be resolved to
    exactly the real, intended product(s) -- this function performs no
    resolution of its own."""
    trace.route = "cart_add"
    unpriced = _unpriced(products)
    if unpriced:
        return _response(
            session_id, None, "cart_add", state, user_message=message,
            assistant_message_override=messages.cart_add_price_unavailable_response(
                message, [(p.name_en, p.name_ar) for p in unpriced],
            ),
        )
    add_quantity = _validated_cart_quantity(action.quantity, default=1)
    if add_quantity is None:
        return _response(
            session_id, None, "cart_add", state, user_message=message,
            assistant_message_override=messages.cart_quantity_missing_response(message),
        )
    had_checkout = activity.checkout_active  # before the mutation below
    activity_store.add_to_cart(session_id, products, quantity=add_quantity)  # also
    # invalidates any open draft
    updated_cart = activity_store.get(session_id).cart
    by_id = {line.product.id: line for line in updated_cart}
    added = [(p.name_en, p.name_ar, by_id[p.id].quantity) for p in products]
    return _response(
        session_id, None, "cart_add", state, user_message=message,
        assistant_message_override=messages.cart_add_confirmation_response(message, added),
        cart=_to_cart_out(updated_cart), checkout_invalidated=had_checkout,
    )


def handle_cart_action(
    cart_action: str, session_id: str, message: str, state: FilterSet, activity: SessionActivity,
    action: Action, activity_store: SessionActivityStore, trace: RequestTrace,
    pending: PendingAction | None = None,
) -> SessionMessageResponse:
    """C3: cart commands -- resolved deterministically; the LLM only
    signals WHICH operation (`cart_action`) and WHICH item(s)
    (selection_ordinals/last/text, reused unchanged via product_selection's
    resolver). Moved verbatim out of src/scenario/orchestrator.py's
    _handle_message_inner(). Exhaustive over Action.cart_action's 5
    literals ("view"/"clear"/"add"/"remove"/"update_quantity") -- the
    caller only calls this when `cart_action is not None`, and every
    branch here returns a real response (never None/falls through)."""
    if cart_action == "view":
        trace.route = "cart_view"
        lines = activity.cart
        reply = messages.cart_view_response(
            message,
            [
                (line.product.name_en, line.product.name_ar, line.quantity,
                 effective_price(line.product.price, line.product.special_price))
                for line in lines
            ],
        )
        return _response(
            session_id, None, "cart_view", state, user_message=message,
            assistant_message_override=reply, cart=_to_cart_out(lines),
        )

    if cart_action == "clear":
        trace.route = "cart_clear"
        had_checkout = activity.checkout_active  # before the mutation below
        activity_store.clear_cart(session_id)  # also invalidates any open draft
        return _response(
            session_id, None, "cart_clear", state, user_message=message,
            assistant_message_override=messages.cart_clear_confirmation_response(message),
            cart=_to_cart_out([]), checkout_invalidated=had_checkout,
        )

    if cart_action == "add":
        trace.route = "cart_add"
        recent_shown = activity_store.recent_shown_products(session_id)
        mid_clarification = pending is not None and pending.kind == "cart_add_pending"
        if mid_clarification:
            # Already mid-clarification: the FROZEN candidate snapshot stays
            # the only authority (same helper the resolver/compact LLM
            # context use), never a freshly re-derived pool, and any
            # further clarification re-freezes these SAME ids.
            candidates = resolve_pending_candidates(activity, activity_store, session_id, pending)
            current_shown, named_pool, authority = candidates, candidates, "pending_candidates"
            frozen_ids = list(pending.shown_product_ids)
        else:
            current_shown, named_pool = activity.last_shown_products, recent_shown
            authority = "last_shown_products+recent_shown"
            frozen_ids = [p.id for p in recent_shown]

        # Generic-reference correction: a bare/pronoun/plural add carries no
        # product name, so a selection_text the LLM copied from the command
        # itself must never be resolved as one -- the resolver only ever
        # sees real reference evidence. (A raw exact/named title in the
        # message is still found by the resolver directly from the raw text.)
        action = _normalize_generic_add(action, message)
        generic = action.generic_reference is not None
        resolve_action = action.model_copy(update={"selection_text": None}) if generic else action
        products, reason = _trace_resolve(trace, authority, current_shown, named_pool, resolve_action, message)
        if reason == "not_found" and not _names_a_product(action):
            # No product evidence at all: a generic cart-add, so the
            # deterministic state -- not the wording -- decides what happens.
            if action.generic_reference == "plural":
                # The whole current shown group (limited to the frozen
                # candidates when mid-clarification) -- the SAME multi-
                # product add path "first and third" already uses.
                frozen = set(frozen_ids) if mid_clarification else None
                group = [p for p in activity.last_shown_products if frozen is None or p.id in frozen]
                if group:
                    return _perform_cart_add(session_id, message, state, group, action, activity, activity_store, trace)
            elif len(current_shown) == 1:
                # A single possible answer needs no clarification.
                return _perform_cart_add(
                    session_id, message, state, current_shown, action, activity, activity_store, trace,
                )
            # 2+ candidates and no way to tell which (or a plural reference
            # with nothing shown): a genuine ambiguity -- ask, backed by
            # durable pending state so the very next answer resolves it
            # deterministically (see src/scenario/pending_actions.py's
            # resolve_pending_cart_add()).
            activity_store.set_pending_action(session_id, "cart_add_pending", shown_product_ids=frozen_ids)
            return _response(
                session_id, None, "cart_add", state, user_message=message,
                assistant_message_override=messages.cart_add_vague_reference_response(message),
            )
        if reason != "ok":
            activity_store.set_pending_action(  # see note above
                session_id, "cart_add_pending", shown_product_ids=frozen_ids,
            )
            return _response(
                session_id, None, "cart_add", state, user_message=message,
                assistant_message_override=_selection_failure_message(
                    message, reason, len(current_shown),
                ),
            )
        return _perform_cart_add(session_id, message, state, products, action, activity, activity_store, trace)

    # cart_action in ("remove", "update_quantity"): both inherently target
    # the CART -- there is no "outside cart context" meaning for either, so
    # both current AND named authority are the cart pool itself
    # (positional references are cart-ORDER here, unlike everywhere else).
    cart_pool = [line.product for line in activity.cart]
    products, reason = _trace_resolve(trace, "cart", cart_pool, cart_pool, action, message)
    if reason != "ok":
        trace.route = f"cart_{cart_action}"
        return _response(
            session_id, None, f"cart_{cart_action}", state, user_message=message,
            assistant_message_override=_selection_failure_message(
                message, reason, len(cart_pool), cart=True,
            ),
        )

    if cart_action == "remove":
        trace.route = "cart_remove"
        had_checkout = activity.checkout_active  # before the mutation below
        removed = activity_store.remove_from_cart(session_id, [p.id for p in products])  # also
        # invalidates any open draft
        return _response(
            session_id, None, "cart_remove", state, user_message=message,
            assistant_message_override=messages.cart_remove_confirmation_response(
                message, [(line.product.name_en, line.product.name_ar) for line in removed],
            ),
            cart=_to_cart_out(activity_store.get(session_id).cart), checkout_invalidated=had_checkout,
        )

    # update_quantity -- must target exactly ONE cart line; no single
    # quantity number could apply to several different lines at once
    # without guessing which applies to which.
    trace.route = "cart_update_quantity"
    if len(products) > 1:
        return _response(
            session_id, None, "cart_update_quantity", state, user_message=message,
            assistant_message_override=messages.cart_selection_ambiguous_response(message),
        )
    # Correction: same authoritative validate_quantity() reuse -- here a
    # quantity is always mandatory (no sensible default), so both "nothing
    # stated" and "an invalid number" ask for clarification identically.
    new_quantity = _validated_cart_quantity(action.quantity, default=None)
    if new_quantity is None:
        return _response(
            session_id, None, "cart_update_quantity", state, user_message=message,
            assistant_message_override=messages.cart_quantity_missing_response(message),
        )
    product = products[0]
    had_checkout = activity.checkout_active  # before the mutation below
    activity_store.set_cart_quantity(session_id, product.id, new_quantity)  # also
    # invalidates any open draft
    return _response(
        session_id, None, "cart_update_quantity", state, user_message=message,
        assistant_message_override=messages.cart_quantity_updated_response(
            message, product.name_en, product.name_ar, new_quantity,
        ),
        cart=_to_cart_out(activity_store.get(session_id).cart), checkout_invalidated=had_checkout,
    )


# ---------------------------------------------------------------------------
# Zero-LLM deterministic fast path (LLM-usage audit task)
# ---------------------------------------------------------------------------

_CART_VIEW_PHRASES_EN = (
    "show my cart", "show cart", "view cart", "view my cart",
    "what's in my cart", "whats in my cart", "what is in my cart",
)
_CART_VIEW_PHRASES_AR = (
    "وريني السلة", "وريني الكارت", "ايه اللي في السلة", "ايه في السلة",
)
_CART_CLEAR_PHRASES_EN = ("clear cart", "clear my cart", "empty cart", "empty my cart")
_CART_CLEAR_PHRASES_AR = ("فضي السلة",)
# Bounded EN/AR phrase sets, deliberately small -- the SAME wording already
# documented as cart_action="view"/"clear" examples in src/understanding/
# llm.py's SYSTEM_PROMPT (plus the equally-bounded "كارت" colloquial variant
# already live-verified to work via the LLM path), never a growing/guessed
# phrase map. Matched as a WHOLE normalized message (not a substring), so
# only an unmistakable, standalone command takes this path -- anything with
# extra wording around it (a real sentence, a question, a qualifier) falls
# through to the normal understand()-driven pipeline exactly as before.

_CART_ADD_VERB_MARKERS_EN = ("add", "put")
_CART_ADD_VERB_MARKERS_AR = ("ضيف", "حط")
_CART_WORD_MARKERS_ADD = ("cart", "سلة", "كارت")  # same convention as
# _CART_MARKERS above, plus "كارت" (live-verified colloquial cart wording,
# e.g. "حط بيبسي في الكارت") -- kept local to this fast path rather than
# widening the shared _CART_MARKERS used elsewhere for cart-mention scoping.


def resolve_deterministic_cart_fast_path(
    session_id: str, message: str, activity: SessionActivity,
    activity_store: SessionActivityStore, trace: RequestTrace, state: FilterSet,
) -> SessionMessageResponse | None:
    """LLM-usage audit task: zero-LLM fast path for cart operations whose
    outcome is ALREADY fully determined by deterministic raw-message
    evidence -- an unmistakable bounded EN/AR command phrase for view/clear,
    or an EXACT full shown/recent product title for add. Returns None
    whenever the message isn't unmistakable enough (no match, or an add
    reference that doesn't resolve to EXACTLY one product), so the caller
    falls through to the normal understand()-driven pipeline exactly as
    before -- this function never guesses, never fuzzy-matches, and never
    infers an unstated product or quantity.

    Reuses the SAME _exact_title_matches() authority product_selection.py's
    resolver itself uses for a raw exact-title mention, the SAME
    SessionActivityStore cart mutation methods, and the SAME response
    builders/messages wording the LLM-driven cart_action path already
    produces -- never a second product resolver or a duplicated mutation
    path. A message this function resolves is recorded in history as
    Intent.NO_OP, the SAME convention already used for the other
    deterministic (zero-LLM) resolvers in src/scenario/pending_actions.py."""
    latin = normalize_latin(message).rstrip("؟?!")
    arabic = normalize_arabic(message).rstrip("؟?!")

    if latin in _CART_VIEW_PHRASES_EN or arabic in _CART_VIEW_PHRASES_AR:
        activity_store.append_history(session_id, message, Intent.NO_OP.value)
        activity_store.increment_turn_count(session_id)
        trace.route = "cart_view"
        lines = activity.cart
        reply = messages.cart_view_response(
            message,
            [
                (line.product.name_en, line.product.name_ar, line.quantity,
                 effective_price(line.product.price, line.product.special_price))
                for line in lines
            ],
        )
        return _response(
            session_id, None, "cart_view", state, user_message=message,
            assistant_message_override=reply, cart=_to_cart_out(lines),
        )

    if latin in _CART_CLEAR_PHRASES_EN or arabic in _CART_CLEAR_PHRASES_AR:
        activity_store.append_history(session_id, message, Intent.NO_OP.value)
        activity_store.increment_turn_count(session_id)
        trace.route = "cart_clear"
        had_checkout = activity.checkout_active  # before the mutation below
        activity_store.clear_cart(session_id)  # also invalidates any open draft
        return _response(
            session_id, None, "cart_clear", state, user_message=message,
            assistant_message_override=messages.cart_clear_confirmation_response(message),
            cart=_to_cart_out([]), checkout_invalidated=had_checkout,
        )

    has_add_verb = (
        any(re.search(rf"\b{re.escape(w)}\b", latin) for w in _CART_ADD_VERB_MARKERS_EN)
        or any(w in arabic for w in _CART_ADD_VERB_MARKERS_AR)
    )
    mentions_cart = any(marker in (latin + arabic) for marker in _CART_WORD_MARKERS_ADD)
    if not (has_add_verb and mentions_cart):
        return None

    pool = activity_store.recent_shown_products(session_id)
    matches = _exact_title_matches(message, pool)
    if len(matches) != 1:
        # 0 matches (no exact title present) or 2+ (genuinely ambiguous) --
        # never guess; defer to the normal LLM-driven resolver, exactly as
        # it already handles both cases.
        return None
    product = matches[0]

    # Conservative: only take this path when every digit run in the raw
    # message is already accounted for by the matched product's OWN name
    # (e.g. "970" in "Pepsi - 970 Ml") -- an extra number outside the title
    # ("add 3 Pepsi - 970 Ml to cart") might be a real stated quantity, which
    # this fast path deliberately never attempts to extract; it defers to
    # the LLM path instead rather than silently defaulting/guessing.
    message_digits = set(re.findall(r"\d+", message.translate(_ARABIC_INDIC_DIGITS)))
    title_digits = set(re.findall(r"\d+", (product.name_en or "").translate(_ARABIC_INDIC_DIGITS)))
    title_digits |= set(re.findall(r"\d+", (product.name_ar or "").translate(_ARABIC_INDIC_DIGITS)))
    if not message_digits <= title_digits:
        return None

    activity_store.append_history(session_id, message, Intent.NO_OP.value)
    activity_store.increment_turn_count(session_id)
    trace.route = "cart_add"
    trace.resolution_authority = "last_shown_products+recent_shown"
    trace.resolution_product_ids = [product.id]
    trace.resolution_reason = "ok"

    unpriced = _unpriced([product])
    if unpriced:
        return _response(
            session_id, None, "cart_add", state, user_message=message,
            assistant_message_override=messages.cart_add_price_unavailable_response(
                message, [(product.name_en, product.name_ar)],
            ),
        )
    had_checkout = activity.checkout_active  # before the mutation below
    activity_store.add_to_cart(session_id, [product], quantity=1)  # also
    # invalidates any open draft
    updated_cart = activity_store.get(session_id).cart
    by_id = {line.product.id: line for line in updated_cart}
    added = [(product.name_en, product.name_ar, by_id[product.id].quantity)]
    return _response(
        session_id, None, "cart_add", state, user_message=message,
        assistant_message_override=messages.cart_add_confirmation_response(message, added),
        cart=_to_cart_out(updated_cart), checkout_invalidated=had_checkout,
    )
