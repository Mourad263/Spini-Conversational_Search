"""FastAPI app exposing System A (the keyword-search baseline, FR-014,
GET /search/keyword) and System B (the conversational endpoint, tasks.md
T030, POST /session/{session_id}/message -- Sprint 4 Part 3) over HTTP.
The two share no route logic: the conversational turn pipeline lives in
src/scenario/orchestrator.py, this module only wires it to FastAPI.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi import FastAPI, HTTPException, Query, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from decimal import Decimal  # noqa: E402

from src.api.schemas import (  # noqa: E402
    CartDeltaRequest, CartMutationResponse, CheckoutOrderLineOut, CheckoutOrderOut, CheckoutSubmitRequest,
    CheckoutSubmitResponse, KeywordSearchResponse, ProductOut, SessionMessageRequest, SessionMessageResponse,
)
from src.scenario.cart_mutation import CartMutationError, apply_cart_delta  # noqa: E402
from src.scenario.checkout_submission import CheckoutSubmissionError, submit_checkout  # noqa: E402
from src.scenario.orchestrator import get_activity_store, get_state_manager, handle_message  # noqa: E402
from src.search_adapter.adapter import InvalidFilterError, SearchAdapter  # noqa: E402
from src.search_engine.keyword_baseline import parse_keyword_query  # noqa: E402

app = FastAPI(title="Conversational Search Baseline -- Keyword Search (System A)")

# Task 3D: the Vite dev server (frontend/) runs on a different origin. Explicit
# local dev origins only -- no wildcard, and no credentials (session_id travels
# in the URL path, not a cookie, so none are needed).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.exception_handler(InvalidFilterError)
async def invalid_filter_handler(request: Request, exc: InvalidFilterError) -> JSONResponse:
    # Same pattern as FastAPI's own Query() validation errors: a clear 422,
    # not a 500, for malformed-but-well-typed input. Not reachable through
    # /search/keyword today (that endpoint only takes free text, and the
    # text-parsing path auto-normalizes an inverted "between X and Y"
    # before this could ever fire) -- this guards any future endpoint
    # (e.g. Sprint 2's conversational route) that constructs a FilterSet
    # from direct structured params.
    return JSONResponse(status_code=422, content={"detail": str(exc)})


def _to_float(v: Decimal | float | None) -> float | None:
    return float(v) if v is not None else None


@app.get("/search/keyword", response_model=KeywordSearchResponse)
def search(
    q: str = Query(..., min_length=1, max_length=500),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> KeywordSearchResponse:
    # NOTE: a matching Pydantic model (KeywordSearchQuery, schemas.py) existed
    # unused here until this fix -- the route previously took bare `q: str`
    # with no length limit and clamped limit/offset manually in Python
    # instead of validating them, so an empty/oversized query never actually
    # got rejected. Found during an adversarial boundary-testing pass, not
    # by reading the code.
    #
    # No dedicated brand= param -- brand is resolved purely from q's free
    # text, the same way category/price already are (fuzzy-matched in
    # parse_keyword_query, exposed only via resolved_brand below).
    filters = parse_keyword_query(q)
    result = SearchAdapter().search(filters, limit=limit, offset=offset)

    return KeywordSearchResponse(
        query=q,
        resolved_category=filters.category,
        resolved_brand=filters.brand,
        price_min=filters.price_min,
        price_max=filters.price_max,
        products=[
            ProductOut(
                id=p["id"],
                name_en=p["name_en"],
                name_ar=p["name_ar"],
                price=_to_float(p["price"]),
                special_price=_to_float(p["special_price"]),
                brand_normalized=p["brand_normalized"],
                url_key=p["url_key"],
                sku=p["sku"],
            )
            for p in result.products
        ],
        total_count=result.total_count,
        zero_result=result.zero_result,
        limit=limit,
        offset=offset,
    )


@app.post("/session/{session_id}/message", response_model=SessionMessageResponse)
def session_message(session_id: str, body: SessionMessageRequest) -> SessionMessageResponse:
    # Oversized/empty `message` is already rejected with a 422 by
    # SessionMessageRequest's Field(min_length=1, max_length=500) before this
    # body ever runs -- understand()/OpenAI is never reached for it.
    return handle_message(
        session_id, body.message,
        state_manager=get_state_manager(),
        activity_store=get_activity_store(),
    )


@app.post("/session/{session_id}/cart/item", response_model=CartMutationResponse)
def cart_item_delta(session_id: str, body: CartDeltaRequest) -> CartMutationResponse:
    # Final UI task: deterministic, 0-LLM cart mutation for UI buttons (the
    # product-card stepper, the cart drawer's +/-) -- the frontend already
    # knows product_id; this never calls handle_message()/understand(), so
    # it can never make a Groq call or fuzzy-resolve a product. Product
    # data is always re-resolved against real catalog data here, never
    # trusted from React (src/scenario/cart_mutation.py).
    try:
        result = apply_cart_delta(
            session_id, body.product_id, body.delta,
            activity_store=get_activity_store(),
        )
    except CartMutationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    return CartMutationResponse(**result)


@app.post("/session/{session_id}/checkout/submit", response_model=CheckoutSubmitResponse)
def checkout_submit(session_id: str, body: CheckoutSubmitRequest) -> CheckoutSubmitResponse:
    # Sprint 6 checkout-submission task: deliberately NOT part of
    # handle_message()'s natural-language turn pipeline -- a structured
    # form POST, zero LLM/search/recommendation/embedding calls. Product
    # identity/price/payment_method are never taken from this request body
    # (see src/scenario/checkout_submission.py's submit_checkout()).
    try:
        result = submit_checkout(
            session_id,
            lines=[(line.product_id, line.quantity) for line in body.lines],
            customer_name=body.customer_name,
            phone=body.phone,
            delivery_address=body.delivery_address,
            state_manager=get_state_manager(),
            activity_store=get_activity_store(),
        )
    except CheckoutSubmissionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    order = result["order"]
    return CheckoutSubmitResponse(
        order=CheckoutOrderOut(
            order_id=order["order_id"],
            lines=[CheckoutOrderLineOut(**line) for line in order["lines"]],
            total_price=order["total_price"],
            payment_method=order["payment_method"],
            status=order["status"],
        ),
        assistant_message=result["assistant_message"],
        cart=result["cart"],
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
