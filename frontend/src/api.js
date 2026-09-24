const BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

// POST /session/{session_id}/message -- the one and only conversational
// endpoint (src/api/routes.py). One user Send = exactly one call here;
// assistant_message/products/recommendations all come back in this same
// response, so nothing else ever needs to be called per turn.
export async function sendMessage(sessionId, message) {
  const response = await fetch(`${BASE_URL}/session/${encodeURIComponent(sessionId)}/message`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });
  if (!response.ok) {
    throw new Error(`Request failed with status ${response.status}`);
  }
  return response.json();
}

// POST /session/{session_id}/cart/item -- the deterministic, 0-LLM cart
// mutation endpoint (src/api/routes.py) UI buttons (ProductCard's stepper,
// the cart drawer's +/-) use instead of the conversational endpoint. Sends
// ONLY product_id + a signed quantity delta; name/price/total always come
// back from the server's own authoritative cart snapshot, never computed
// here.
export async function mutateCartItem(sessionId, productId, delta) {
  const response = await fetch(`${BASE_URL}/session/${encodeURIComponent(sessionId)}/cart/item`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ product_id: productId, delta }),
  });
  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`;
    try {
      const body = await response.json();
      if (typeof body?.detail === "string") detail = body.detail;
    } catch {
      // non-JSON error body -- keep the generic message
    }
    throw new Error(detail);
  }
  return response.json();
}

// POST /session/{session_id}/checkout/submit -- the structured DEMO-order
// submission endpoint (src/api/routes.py). Sends ONLY `lines` (product_id +
// quantity per line -- product_id must be one of the active checkout's own
// server-issued line ids) and the three customer-supplied fields; product
// identity/price/payment_method/order_id are server-authoritative and are
// never part of this request body.
export async function submitCheckout(sessionId, { lines, customer_name, phone, delivery_address }) {
  const response = await fetch(`${BASE_URL}/session/${encodeURIComponent(sessionId)}/checkout/submit`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ lines, customer_name, phone, delivery_address }),
  });
  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`;
    try {
      const body = await response.json();
      if (typeof body?.detail === "string") detail = body.detail;
    } catch {
      // non-JSON error body -- keep the generic message
    }
    throw new Error(detail);
  }
  return response.json();
}
