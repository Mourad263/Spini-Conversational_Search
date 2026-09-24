"""Orchestrator organization refactor: checkout-specific logic, moved
verbatim out of src/scenario/orchestrator.py. `_checkout_from_lines()` is
the ONE convergence point every checkout flow (direct single-product,
cart-all, cart-subset) reaches once its line items are already
deterministically resolved by src/scenario/product_selection.py -- never a
second checkout implementation per flow. `_raw_message_supports_checkout_
all()` is the deterministic "all"/"everything" marker check that turns a
bare cart checkout into an immediate checkout instead of a clarification.
"""

from __future__ import annotations

from src.api.schemas import CheckoutFields, CheckoutForm, CheckoutProductOut, SessionMessageResponse
from src.ingestion.text_normalize import normalize_arabic, normalize_latin
from src.response import messages
from src.scenario.cart_flow import _unpriced
from src.scenario.response_builder import _response
from src.scenario.session_activity import CartLine, SessionActivityStore
from src.scenario.trace import RequestTrace
from src.search_adapter.adapter import FilterSet

_CHECKOUT_ALL_MARKERS_EN = ("all", "everything")
_CHECKOUT_ALL_MARKERS_AR = ("كله", "الكل", "كلها")  # colloquial Egyptian
# "checkout all" forms ("حاسب على كله", "اشتري كل اللي في السلة")


def _raw_message_supports_checkout_all(raw_message: str) -> bool:
    """True only when the user's own words explicitly say "all"/"everything"
    (or the Arabic equivalents) -- what turns a bare cart checkout into an
    immediate CHECKOUT ALL instead of the CASE 3 clarification."""
    latin = normalize_latin(raw_message)
    arabic = normalize_arabic(raw_message)
    return (
        any(marker in latin.split() for marker in _CHECKOUT_ALL_MARKERS_EN)
        or any(marker in arabic for marker in _CHECKOUT_ALL_MARKERS_AR)
    )


def _checkout_from_lines(
    session_id: str, message: str, state: FilterSet, lines: list[CartLine],
    activity_store: SessionActivityStore, trace: RequestTrace, *, cart_ids: list[int],
) -> SessionMessageResponse:
    """Cart/direct-checkout architecture task: the ONE convergence point
    every checkout flow (direct single-product, cart-all, cart-subset)
    reaches once its line items are already deterministically resolved --
    never a second checkout implementation per flow. `lines` always come
    from real shown-product or cart authority (never LLM-chosen, never a
    fresh catalog lookup); `cart_ids` is [] for a direct checkout (cart
    untouched on success) or the cart product ids this draft came from (so
    checkout_submission.py knows which lines to remove after a successful
    purchase -- see SessionActivity.checkout_cart_ids)."""
    unpriced_products = _unpriced([line.product for line in lines])
    if unpriced_products:
        trace.route = "checkout_price_unavailable"
        reply = messages.checkout_price_unavailable_response(
            message, [(p.name_en, p.name_ar) for p in unpriced_products],
        )
        return _response(
            session_id, None, "checkout_clarification", state,
            user_message=message, assistant_message_override=reply,
        )
    activity_store.set_selected_products(session_id, lines)
    activity_store.start_checkout(session_id, cart_ids=cart_ids)
    form = CheckoutForm(
        lines=[
            CheckoutProductOut(
                id=line.product.id, name_en=line.product.name_en, name_ar=line.product.name_ar,
                price=line.product.price, special_price=line.product.special_price, quantity=line.quantity,
            )
            for line in lines
        ],
        fields=CheckoutFields(),
    )
    reply = messages.checkout_started_response(
        message, [(line.product.name_en, line.product.name_ar, line.quantity) for line in lines],
    )
    # Correction: set HERE, the one place a checkout draft actually opens
    # successfully -- never by a caller setting a premature "checkout_start"
    # default before resolution is even attempted (real bug: that default
    # used to leak into the trace on an ambiguous/failed-resolution
    # clarification response too, since nothing reset it on that path).
    trace.route = "checkout_start"
    return _response(
        session_id, None, "checkout_start", state,
        user_message=message, assistant_message_override=reply, checkout_form=form,
    )
