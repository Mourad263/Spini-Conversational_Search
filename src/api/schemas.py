"""FastAPI request/response Pydantic models -- the third of the three
Pydantic usage sites named in plan.md (LLM Action schema and the
conversational session response below are the other two)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ProductOut(BaseModel):
    id: int
    name_en: str
    name_ar: str
    price: float | None
    special_price: float | None
    brand_normalized: str | None
    url_key: str | None
    sku: str | None


class KeywordSearchResponse(BaseModel):
    query: str
    resolved_category: str | None
    resolved_brand: str | None
    price_min: float | None
    price_max: float | None
    products: list[ProductOut]
    total_count: int
    zero_result: bool
    limit: int
    offset: int


class CheckoutProductOut(BaseModel):
    """Sprint 6 checkout task: the REAL selected product's data for the
    checkout form -- deliberately a small subset of ProductOut (no
    brand_normalized/url_key/sku: not needed for this draft), sourced from
    src/scenario/session_activity.py's DisplayedProduct snapshot, never a
    fresh SearchAdapter/catalog lookup."""

    id: int
    name_en: str
    name_ar: str
    price: float | None
    special_price: float | None
    quantity: int = 1  # Cart/direct-checkout architecture task: 1 for a DIRECT
    # checkout line; copied from the cart line's own quantity for a cart-sourced
    # (all/subset) checkout -- the checkout confirmation always states the real
    # quantity before the form, never leaving the user to guess it.


class CartLineOut(BaseModel):
    """Cart/direct-checkout architecture task: one durable cart line for the
    frontend's cart display -- unit_price/line_total are always the real,
    known src/pricing.py effective_price() (correction: an unpriced
    product can no longer enter the cart at all, so these are never null
    here -- see orchestrator.py's _unpriced())."""

    id: int
    name_en: str
    name_ar: str
    quantity: int
    unit_price: float
    line_total: float


class CartOut(BaseModel):
    """Cart/direct-checkout architecture task: the durable cart's current
    snapshot -- server-authoritative, included on cart-relevant turns only
    (SessionMessageResponse.cart is null on an unrelated turn; the cart
    itself is unchanged then, so the frontend keeps showing its last known
    snapshot rather than being told to blank it)."""

    lines: list[CartLineOut]
    total: float  # the REAL full cart total -- an unpriced product can never enter
    # the cart at all (see orchestrator.py's _unpriced()), so every line always has a
    # real, known unit_price and this is never a known-price subset.


class CheckoutFields(BaseModel):
    """Sprint 6 checkout task: the fields the (future) frontend form will
    collect -- deliberately all null/empty here, since nothing has been
    submitted yet. Quantity is per LINE (CheckoutSubmitRequest.lines), not
    here -- this only covers the one-per-order customer fields."""

    customer_name: str | None = None
    phone: str | None = None
    delivery_address: str | None = None


class CheckoutForm(BaseModel):
    """Sprint 6 checkout task; cart/direct-checkout architecture task
    (correction): a checkout DRAFT description for the frontend to render
    later -- never an order. `lines` is server-authoritative (sourced only
    from SessionActivity.selected_products, never a client value) and
    always non-empty -- exactly ONE entry for a DIRECT checkout (from
    shown-product authority; 2+ named products there become a cart
    proposal instead, never a multi-line direct draft), or one entry per
    cart line for a CART checkout (all lines, or a resolved subset).
    payment_method is always cash_on_delivery."""

    lines: list[CheckoutProductOut]
    fields: CheckoutFields = Field(default_factory=CheckoutFields)
    payment_method: Literal["cash_on_delivery"] = "cash_on_delivery"


class CheckoutLineIn(BaseModel):
    """Sprint 6 checkout-submission task (generalized): the ONLY per-line
    fields a client may submit -- `product_id` must be one of the active
    checkout's own CheckoutForm.lines ids (never an arbitrary catalog id;
    see src/scenario/checkout_submission.py), `quantity`'s Field(ge=1) is a
    shallow defense-in-depth check backed by demo_order.py's
    validate_quantity()."""

    product_id: int
    quantity: int = Field(..., ge=1)


class CheckoutSubmitRequest(BaseModel):
    """Sprint 6 checkout-submission task: the ONLY fields a client may
    submit. Deliberately no product name/price/payment_method/order
    status/order_id -- those are always server-controlled (see
    src/checkout/demo_order.py). `lines` must name every product from the
    active checkout's selection, each exactly once -- never a subset
    (silently dropping a line), never an unknown id."""

    lines: list[CheckoutLineIn] = Field(..., min_length=1)
    customer_name: str
    phone: str
    delivery_address: str


class CheckoutOrderLineOut(BaseModel):
    """Sprint 6 checkout-submission task (generalized): one persisted
    order line, echoed back in the submission response -- id/name/
    quantity/unit_price/line_total, all server-computed."""

    id: int
    name_en: str
    name_ar: str
    quantity: int
    unit_price: float
    line_total: float


class CheckoutOrderOut(BaseModel):
    """Sprint 6 checkout-submission task: the persisted DemoOrder's public
    summary -- deliberately omits customer name/phone/delivery_address
    (PII lives only in the saved JSON file, never in an API response)."""

    order_id: str
    lines: list[CheckoutOrderLineOut]
    total_price: float
    payment_method: Literal["cash_on_delivery"]
    status: Literal["submitted"]


class CheckoutSubmitResponse(BaseModel):
    """Sprint 6 checkout-submission task: the full response for a
    successful POST to the checkout-submission endpoint."""

    order: CheckoutOrderOut
    assistant_message: str
    cart: CartOut  # Cart/direct-checkout architecture task: the cart's snapshot
    # AFTER this purchase -- unchanged for a direct checkout, purchased lines
    # removed for a cart checkout (all or subset; see checkout_submission.py).
    # Always present (never null) so the frontend can sync its display right
    # after a successful order without a second round trip.


class CartDeltaRequest(BaseModel):
    """Final UI task: the ONLY fields a UI cart-mutation button may submit
    -- product_id identifies the exact line, delta is the signed quantity
    change (+1 for the "Add to cart"/"+" controls, -1 for "-"). Name/price/
    quantity stay server-authoritative (src/scenario/cart_mutation.py
    always re-resolves product_id against real catalog data)."""

    product_id: int
    delta: int


class CartMutationResponse(BaseModel):
    """Same authoritative cart snapshot shape as SessionMessageResponse.cart,
    plus whether this mutation invalidated a previously-open checkout draft
    -- same semantics as SessionMessageResponse.checkout_invalidated."""

    cart: CartOut
    checkout_invalidated: bool


class SessionMessageRequest(BaseModel):
    # Enforced here (Pydantic/FastAPI layer) so an oversized message is
    # rejected with a 422 before src/scenario/orchestrator.py -- and
    # therefore understand()/OpenAI -- is ever reached (Sprint 4 Part 3),
    # same bound as /search/keyword's `q` (routes.py).
    message: str = Field(..., min_length=1, max_length=500)


class SessionMessageResponse(BaseModel):
    """Sprint 4 Part 3 conversational turn response. resolved_category/
    resolved_brand/price_min/price_max always reflect the CURRENT
    canonical StateManager state after this turn (never only the current
    turn's own mentioned fields) -- see src/scenario/orchestrator.py.
    effective_price_min/effective_price_max are populated ONLY during an
    automatic recommendation-exploration price-band turn; they are never
    the user's canonical budget and are never written back into
    StateManager."""

    session_id: str
    message: str | None  # deterministic wording (off-topic/clarification/
    # acknowledgment/zero-result/...); null for an ordinary successful
    # search turn with nothing extra to say
    intent: str  # the effective intent this turn resolved to, after
    # deterministic normalization/repair (src/scenario/decision.py)
    resolved_category: str | None
    resolved_brand: str | None
    price_min: float | None
    price_max: float | None
    effective_price_min: float | None = None
    effective_price_max: float | None = None
    products: list[ProductOut]
    total_count: int
    zero_result: bool
    likely_out_of_catalog: bool = False
    blocked: bool = False  # true only for a rapid-repeat-blocked request
    recommendations: list[ProductOut] = Field(default_factory=list)  # Sprint 6
    # recommendation task: ML-generated (K=128 content-clustering) catalog-backed
    # ALTERNATIVES, distinct from `products` (the direct search-result set).
    # Populated only when action.recommendation_request is true AND real search
    # evidence + an active filter target exist; [] otherwise -- never a fallback
    # or a relaxed/looser result set. See src/recommendation/service.py.
    assistant_message: str = ""  # Task 3A: a natural, ALWAYS-populated bilingual
    # (EN/AR) conversational reply -- separate from `message` above, whose
    # existing (often-null, supplementary-note) meaning is unchanged. Built
    # deterministically from already-known signals only, never a catalog fact
    # the system doesn't actually have. See src/response/messages.py.
    checkout_form: CheckoutForm | None = None  # Sprint 6 checkout task: populated ONLY
    # on a successful checkout initiation (a real selected_product exists); null on
    # every other turn, including a checkout request with nothing selected yet. A
    # DRAFT for the frontend to render later, never an order -- nothing is submitted here.
    cart: CartOut | None = None  # Cart/direct-checkout architecture task: the cart's
    # current snapshot, populated only on a cart-relevant turn (view/add/remove/clear/
    # update-quantity, a cart-sourced checkout, or a successful cart-checkout
    # completion) -- see CartOut. Null on an unrelated turn (the cart didn't change).
    checkout_invalidated: bool = False  # Cart/direct-checkout architecture task
    # (correction, generalized): true whenever this turn caused ANY server-side
    # invalidation of a PREVIOUSLY-OPEN checkout draft -- not only a cart mutation
    # (add/remove/clear/update-quantity, whose session_activity.py methods call
    # clear_checkout() as a side effect), but also RESET and a normal new search/
    # topic switch (orchestrator.py's F branch), both of which also call
    # clear_checkout(). Reflects a REAL invalidation only (orchestrator.py
    # captures activity.checkout_active BEFORE each clearing call, never a blanket
    # true) -- false when there was no open draft to invalidate in the first
    # place. The frontend must drop any checkout_form still showing from an
    # earlier turn whenever this is true (it may reference products/quantities
    # that no longer match reality).
