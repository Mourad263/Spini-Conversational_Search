"""Final UI + deterministic cart backend task: the structured, button-driven
cart mutation -- deliberately NOT part of src/scenario/orchestrator.py's
handle_message() turn pipeline (no natural-language message, no
understand()/Groq call, no fuzzy product resolution). A UI button already
knows the exact product_id; this module only resolves/validates that id
against real catalog data (src/search_engine/query.py's product_by_id())
and applies a signed quantity delta via the SAME SessionActivityStore
cart-mutation methods the chat-command path already uses
(add_to_cart/remove_from_cart/set_cart_quantity) -- never a second cart
implementation. Each of those methods already invalidates any open
checkout draft as its own first step, so this module never re-implements
that either.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.scenario.cart_flow import _to_cart_out, _unpriced  # noqa: E402
from src.scenario.session_activity import DisplayedProduct, SessionActivityStore  # noqa: E402
from src.search_engine.query import product_by_id  # noqa: E402


class CartMutationError(Exception):
    """Any reason a button-driven cart mutation was rejected -- an unknown
    product_id, or a known product with no real known price (the SAME
    purchase-authority rule chat-driven cart add already enforces -- see
    src/scenario/cart_flow.py's _unpriced())."""

    def __init__(self, message: str, status_code: int = 422) -> None:
        super().__init__(message)
        self.status_code = status_code


def _to_float(v) -> float | None:
    return float(v) if v is not None else None


def apply_cart_delta(
    session_id: str, product_id: int, delta: int, *,
    activity_store: SessionActivityStore, now: float | None = None,
) -> dict:
    """Applies a signed quantity `delta` to ONE cart line, identified ONLY
    by product_id -- never a name/price/quantity trusted from the client.
    `delta > 0` on a product not yet in the cart adds it (after a real
    catalog lookup + the same unpriced-product guard chat cart-add uses);
    `delta < 0` reaching zero or below removes the line, matching existing
    server cart semantics exactly (never a separate zero-quantity state).
    `delta == 0` is a harmless no-op. Returns
    {"cart": CartOut, "checkout_invalidated": bool} -- the SAME snapshot
    shape as SessionMessageResponse.cart; `checkout_invalidated` reflects
    whether a checkout draft was open before this call (each reused
    SessionActivityStore method already clears it as its own first step,
    exactly like the chat-command path)."""
    now = time.monotonic() if now is None else now
    activity = activity_store.get(session_id, now=now)  # lazily resets if expired
    had_checkout = activity.checkout_active

    if delta != 0:
        by_id = {line.product.id: line for line in activity.cart}
        existing = by_id.get(product_id)

        if delta > 0:
            if existing is not None:
                activity_store.add_to_cart(session_id, [existing.product], quantity=delta)
            else:
                row = product_by_id(product_id)
                if row is None:
                    raise CartMutationError("Unknown product_id.", status_code=404)
                product = DisplayedProduct(
                    id=row["id"], name_en=row["name_en"], name_ar=row["name_ar"],
                    brand_normalized=row["brand_normalized"],
                    price=_to_float(row["price"]), special_price=_to_float(row["special_price"]),
                )
                if _unpriced([product]):
                    raise CartMutationError("Product has no known price.", status_code=422)
                activity_store.add_to_cart(session_id, [product], quantity=delta)
        else:  # delta < 0
            if existing is None:
                pass  # nothing to remove -- a harmless no-op, never an error
            elif existing.quantity + delta <= 0:
                activity_store.remove_from_cart(session_id, [product_id])
            else:
                activity_store.set_cart_quantity(session_id, product_id, existing.quantity + delta)

    updated_cart = activity_store.get(session_id).cart
    return {"cart": _to_cart_out(updated_cart), "checkout_invalidated": had_checkout}
