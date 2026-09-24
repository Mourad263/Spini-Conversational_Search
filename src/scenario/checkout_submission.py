"""Sprint 6 checkout-submission task: the structured, in-chat checkout
FORM submission -- deliberately NOT part of src/scenario/orchestrator.py's
handle_message() turn pipeline (no natural-language message, no
understand() call, no history/turn_count bookkeeping). Session-aware glue
between the pure src/checkout/demo_order.py (validation + JSON
persistence) and the existing SessionActivity/StateManager lifecycle.

Order of operations (never reordered -- see module docstring in
demo_order.py and the checkout task's own spec):
    validate session invariants (checkout_active, a real selected snapshot)
        -> validate submitted fields
        -> build the server-authoritative DemoOrder
        -> validate it against schemas/demo_order.schema.json
        -> atomically persist the JSON file
        -> ONLY THEN clear checkout_active
        -> return a deterministic confirmation

Any failure at any step raises CheckoutSubmissionError and leaves
checkout_active untouched (still True when it was) -- nothing is ever
half-applied.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.checkout.demo_order import (  # noqa: E402
    DemoOrderValidationError,
    ProductSnapshot,
    build_demo_order,
    save_demo_order,
    validate_customer_name,
    validate_delivery_address,
    validate_lines,
    validate_phone,
)
from src.ingestion.text_normalize import is_arabic_dominant  # noqa: E402
from src.response import messages  # noqa: E402
from src.scenario.cart_flow import _to_cart_out  # noqa: E402
from src.scenario.session_activity import SessionActivityStore  # noqa: E402
from src.scenario.trace import RequestTrace, print_human_trace, spini_trace_enabled  # noqa: E402
from src.state_manager.state_manager import StateManager  # noqa: E402


class CheckoutSubmissionError(Exception):
    """Any reason a checkout submission was rejected -- invalid session
    state, invalid fields, or a persistence failure. `status_code`
    distinguishes a client mistake (422) from an internal failure (500);
    in both cases no order file is left behind and checkout_active is
    never cleared."""

    def __init__(self, message: str, status_code: int = 422) -> None:
        super().__init__(message)
        self.status_code = status_code


def submit_checkout(
    session_id: str,
    *,
    lines: list[tuple[int, int]],
    customer_name: str,
    phone: str,
    delivery_address: str,
    state_manager: StateManager,
    activity_store: SessionActivityStore,
    orders_dir: Path | None = None,
    schema_path: Path | None = None,
    now: float | None = None,
) -> dict:
    """Public entry point -- unchanged behavior/signature/exceptions.
    Instructor-demo observability task: wraps _submit_checkout_impl() with
    one RequestTrace, mirroring src/scenario/orchestrator.py's own
    handle_message()/_handle_message_inner() split. Tracing never
    influences validation/persistence (every trace read below is read-
    only, taken with no `now`, so it never affects expiry/TTL) and never
    logs PII -- see src/scenario/trace.py's own SECURITY note; only
    presence/validity booleans and counts are recorded here, never the raw
    customer_name/phone/delivery_address."""
    trace = RequestTrace(session_id=session_id)
    trace.route = "checkout_submission"
    trace.message_text = "<checkout form submission>"
    start = time.monotonic()
    before = activity_store.get(session_id)
    trace.shown_products_before = len(before.last_shown_products)
    trace.cart_items_before = len(before.cart)
    trace.checkout_draft_before = before.checkout_active
    trace.checkout_submission_items = len(lines)
    trace.checkout_submission_name_present = bool(customer_name and customer_name.strip())
    trace.checkout_submission_address_present = bool(delivery_address and delivery_address.strip())
    try:
        # Reuses the SAME real validator _submit_checkout_impl() itself
        # calls below -- never a second, duplicated phone-format check --
        # purely to record a boolean for the trace; its (non-PII) return
        # value is discarded.
        validate_phone(phone)
        trace.checkout_submission_phone_valid = True
    except DemoOrderValidationError:
        trace.checkout_submission_phone_valid = False
    try:
        result = _submit_checkout_impl(
            session_id, lines=lines, customer_name=customer_name, phone=phone,
            delivery_address=delivery_address, state_manager=state_manager,
            activity_store=activity_store, orders_dir=orders_dir, schema_path=schema_path, now=now,
        )
        trace.response_status = result["order"]["status"]
        trace.response_checkout_form = False
        return result
    except CheckoutSubmissionError as exc:
        trace.response_status = f"error({exc.status_code})"
        raise
    finally:
        trace.total_latency_ms = round((time.monotonic() - start) * 1000, 2)
        after = activity_store.get(session_id)
        trace.cart_items_after = len(after.cart)
        trace.checkout_draft_after = after.checkout_active
        trace.checkout_items_after = len(after.selected_products)
        if spini_trace_enabled():
            print_human_trace(trace)


def _submit_checkout_impl(
    session_id: str,
    *,
    lines: list[tuple[int, int]],
    customer_name: str,
    phone: str,
    delivery_address: str,
    state_manager: StateManager,
    activity_store: SessionActivityStore,
    orders_dir: Path | None = None,
    schema_path: Path | None = None,
    now: float | None = None,
) -> dict:
    """`lines` is (product_id, quantity) pairs, exactly as submitted.
    Returns {"order": {...small summary...}, "assistant_message": "..."}
    on success. Raises CheckoutSubmissionError on any failure -- see module
    docstring for the exact ordering guarantee. Zero LLM/search/
    recommendation/embedding calls: this function touches none of them."""
    now = time.monotonic() if now is None else now
    if activity_store.is_expired(session_id, now):
        state_manager.clear(session_id)
    activity = activity_store.get(session_id, now=now)

    # Section 7: product authority. An expired/never-started/already-
    # cleared session naturally has checkout_active=False here -- no
    # special-casing needed, the EXISTING lazy-expiry mechanism already
    # covers it.
    if not activity.checkout_active or not activity.selected_products:
        raise CheckoutSubmissionError("No active checkout for this session.")
    by_id = {line.product.id: line.product for line in activity.selected_products}

    try:
        matched_lines = validate_lines(lines, set(by_id))
        customer_name = validate_customer_name(customer_name)
        phone = validate_phone(phone)
        delivery_address = validate_delivery_address(delivery_address)
    except DemoOrderValidationError as exc:
        raise CheckoutSubmissionError(str(exc), status_code=422) from exc

    order_lines = []
    for product_id, quantity in matched_lines:
        snapshot = by_id[product_id]
        order_lines.append((
            ProductSnapshot(
                id=snapshot.id, name_en=snapshot.name_en, name_ar=snapshot.name_ar,
                price=snapshot.price, special_price=snapshot.special_price,
            ),
            quantity,
        ))
    try:
        order = build_demo_order(session_id, order_lines, customer_name, phone, delivery_address)
    except DemoOrderValidationError as exc:
        raise CheckoutSubmissionError(str(exc), status_code=422) from exc

    try:
        save_demo_order(order, orders_dir=orders_dir, schema_path=schema_path)
    except Exception as exc:
        # jsonschema.ValidationError, OSError, or DemoOrderValidationError
        # (e.g. an order-id collision) -- none of these are the customer's
        # fault, and none may ever produce a fake success or a final file.
        raise CheckoutSubmissionError(f"Order could not be saved: {exc}", status_code=500) from exc

    # ONLY reached after successful persistence -- see module docstring.
    # The purchase is now COMPLETE: the canonical shopping FilterSet this
    # order was built from is fulfilled, not a still-active search a later
    # vague turn ("I want to buy another thing") should inherit -- left
    # untouched, that stale state made such a turn silently re-search (or
    # NO_OP-clarify against) the already-bought item's own old context
    # instead of starting a genuinely fresh one. Same "drop the FilterSet
    # entirely" call session expiry already uses (src/scenario/
    # orchestrator.py); RESET reuses apply() instead only because it has a
    # real Action to pass through -- this endpoint never does.
    state_manager.clear(session_id)
    # Cart/direct-checkout architecture task: read checkout_cart_ids BEFORE
    # clear_checkout() wipes it below -- non-empty means this was a cart
    # checkout (all or subset), so ONLY those purchased lines are removed,
    # and ONLY now, after persistence actually succeeded (never at
    # initiation -- see module docstring's ordering guarantee). Empty means
    # a DIRECT checkout -- the cart is left completely untouched.
    if activity.checkout_cart_ids:
        activity_store.remove_from_cart(session_id, activity.checkout_cart_ids)
    activity_store.clear_checkout(session_id)
    activity_store.clear_recent_shown_products(session_id)  # cross-turn checkout task:
    # a completed purchase must not leave an earlier turn's product still
    # checkout-selectable by name into the next, unrelated shopping flow

    # Language comes from the CONVERSATION, never customer-submitted PII
    # (name/address are free-text and may be written in either script for
    # all kinds of reasons unrelated to which language the chat itself is
    # in). activity.history[-1] is the same raw-message signal every other
    # response in this project already keys its language off of (see
    # is_arabic_dominant()'s callers in src/response/messages.py) -- and is
    # guaranteed non-empty here: checkout_active only ever becomes True via
    # a real conversational turn that already appended to history.
    arabic = is_arabic_dominant(activity.history[-1])
    assistant_message = messages.checkout_order_submitted_response(
        arabic, [(p["name_en"], p["name_ar"], p["quantity"]) for p in order["products"]],
    )
    return {
        "order": {
            "order_id": order["order_id"],
            "lines": [
                {
                    "id": p["id"], "name_en": p["name_en"], "name_ar": p["name_ar"],
                    "quantity": p["quantity"], "unit_price": p["unit_price"], "line_total": p["line_total"],
                }
                for p in order["products"]
            ],
            "total_price": order["total_price"],
            "payment_method": order["payment_method"],
            "status": order["status"],
        },
        "assistant_message": assistant_message,
        # .model_dump() -- keeps this dict JSON-serializable on its own
        # (matches every other key here); routes.py's CheckoutSubmitResponse
        # still accepts the plain dict for its `cart: CartOut` field via
        # normal pydantic coercion.
        "cart": _to_cart_out(activity_store.get(session_id).cart).model_dump(),
    }
