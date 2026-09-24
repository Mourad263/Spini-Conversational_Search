"""Orchestrator organization refactor: the response-construction helpers
every branch of src/scenario/orchestrator.py's turn pipeline converges on --
shaping a raw catalog/recommendation dict into a typed ProductOut/
DisplayedProduct, and building the final SessionMessageResponse. Pure
functions only (no session/state mutation) -- moved here verbatim from
orchestrator.py so the other extracted flow modules (product_selection,
cart_flow, checkout_flow, pending_actions) can build a response without
importing orchestrator.py itself (which would be circular, since
orchestrator.py delegates to those modules).
"""

from __future__ import annotations

from src.api.schemas import CartOut, CheckoutForm, ProductOut, SessionMessageResponse
from src.response import messages
from src.scenario.session_activity import DisplayedProduct
from src.search_adapter.adapter import FilterSet


def _to_float(v) -> float | None:
    return float(v) if v is not None else None


def _to_product_out(p: dict) -> ProductOut:
    return ProductOut(
        id=p["id"],
        name_en=p["name_en"],
        name_ar=p["name_ar"],
        price=_to_float(p["price"]),
        special_price=_to_float(p["special_price"]),
        brand_normalized=p["brand_normalized"],
        url_key=p["url_key"],
        sku=p["sku"],
    )


def _to_displayed_products(products: list[dict]) -> list[DisplayedProduct]:
    """Sprint 6 product-selection task: the smallest real subset needed to
    resolve a later reference, built directly from the SAME dict rows
    already fetched this turn -- no new DB query. price/special_price
    (Sprint 6 checkout task) ride along too, for the same reason."""
    return [
        DisplayedProduct(
            id=p["id"], name_en=p["name_en"], name_ar=p["name_ar"],
            brand_normalized=p["brand_normalized"],
            price=_to_float(p["price"]), special_price=_to_float(p["special_price"]),
        )
        for p in products
    ]


def _response(
    session_id: str,
    message: str | None,
    intent: str,
    state: FilterSet,
    *,
    user_message: str,
    products: list[dict] | None = None,
    total_count: int = 0,
    zero_result: bool = False,
    likely_out_of_catalog: bool = False,
    blocked: bool = False,
    effective_price_min: float | None = None,
    effective_price_max: float | None = None,
    recommendations: list[dict] | None = None,
    recommendation_request: bool = False,
    needs_clarification: bool = False,
    assistant_message_override: str | None = None,
    checkout_form: CheckoutForm | None = None,
    cart: CartOut | None = None,
    checkout_invalidated: bool = False,
) -> SessionMessageResponse:
    recommendations = recommendations or []
    assistant_message = assistant_message_override if assistant_message_override is not None else (
        messages.build_assistant_message(
            user_message=user_message, intent=intent, recommendation_request=recommendation_request,
            recommendations_count=len(recommendations), zero_result=zero_result,
            likely_out_of_catalog=likely_out_of_catalog, blocked=blocked,
            needs_clarification=needs_clarification,
        )
    )
    return SessionMessageResponse(
        session_id=session_id,
        message=message,
        intent=intent,
        resolved_category=state.category,
        resolved_brand=state.brand,
        price_min=state.price_min,
        price_max=state.price_max,
        effective_price_min=effective_price_min,
        effective_price_max=effective_price_max,
        products=[_to_product_out(p) for p in (products or [])],
        total_count=total_count,
        zero_result=zero_result,
        likely_out_of_catalog=likely_out_of_catalog,
        blocked=blocked,
        recommendations=[_to_product_out(p) for p in recommendations],
        assistant_message=assistant_message,
        checkout_form=checkout_form,
        cart=cart,
        checkout_invalidated=checkout_invalidated,
    )
