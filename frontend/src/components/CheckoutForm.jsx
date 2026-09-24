import { useState } from "react";
import { submitCheckout } from "../api.js";
import ProductCard from "./ProductCard.jsx";

const LABELS = {
  en: {
    title: "Checkout",
    quantity: "Quantity",
    name: "Name",
    phone: "Phone",
    address: "Delivery Address",
    payment: "Payment Method",
    cod: "Cash on Delivery",
    submit: "Place Order",
    submitting: "Placing order...",
  },
  ar: {
    title: "إتمام الطلب",
    quantity: "الكمية",
    name: "الاسم",
    phone: "رقم الهاتف",
    address: "عنوان التوصيل",
    payment: "طريقة الدفع",
    cod: "الدفع عند الاستلام",
    submit: "تأكيد الطلب",
    submitting: "جاري إرسال الطلب...",
  },
};

// Light UX validation only -- the backend (src/checkout/demo_order.py) is
// the sole authority on real field validity (phone shape, min address
// length, quantities matching the active selection, etc). This just stops
// an obviously-empty/incomplete submit round trip.
function isFormFilled({ quantities, customerName, phone, deliveryAddress }) {
  return (
    Object.values(quantities).every((q) => Number(q) >= 1) &&
    customerName.trim() !== "" && phone.trim() !== "" && deliveryAddress.trim() !== ""
  );
}

export default function CheckoutForm({ form, sessionId, arabic, onSuccess }) {
  // One quantity per server-authoritative line (form.lines), keyed by
  // product id -- the server never accepts a client-invented product id,
  // so this map can only ever describe THOSE products. Pre-filled from the
  // line's own server-resolved quantity (cart/direct-checkout architecture
  // task: 1 for a direct checkout, the real cart quantity for a cart
  // checkout) -- still freely editable before submitting.
  const [quantities, setQuantities] = useState(
    () => Object.fromEntries(form.lines.map((p) => [p.id, p.quantity ?? 1])),
  );
  const [customerName, setCustomerName] = useState("");
  const [phone, setPhone] = useState("");
  const [deliveryAddress, setDeliveryAddress] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);

  const t = arabic ? LABELS.ar : LABELS.en;
  const fields = { quantities, customerName, phone, deliveryAddress };

  async function handleSubmit(e) {
    e.preventDefault();
    if (submitting || !isFormFilled(fields)) return; // guards duplicate/incomplete submits

    setSubmitting(true);
    setError(null);
    try {
      const result = await submitCheckout(sessionId, {
        lines: form.lines.map((p) => ({ product_id: p.id, quantity: Number(quantities[p.id]) })),
        customer_name: customerName.trim(),
        phone: phone.trim(),
        delivery_address: deliveryAddress.trim(),
      });
      onSuccess(result);
      // No further state update here -- the parent removes this form from
      // the conversation on success, so this component unmounts.
    } catch (err) {
      setError(err.message || "Order could not be submitted.");
      setSubmitting(false);
    }
  }

  return (
    <form className="checkout-form" dir={arabic ? "rtl" : "ltr"} onSubmit={handleSubmit}>
      <div className="checkout-form-title">{t.title}</div>

      {form.lines.map((product) => (
        <div className="checkout-line" key={product.id}>
          <ProductCard product={product} />
          <label className="checkout-field">
            {t.quantity}
            <input
              type="number"
              min="1"
              step="1"
              value={quantities[product.id]}
              onChange={(e) => setQuantities((prev) => ({ ...prev, [product.id]: e.target.value }))}
              disabled={submitting}
              required
            />
          </label>
        </div>
      ))}

      <label className="checkout-field">
        {t.name}
        <input
          type="text"
          value={customerName}
          onChange={(e) => setCustomerName(e.target.value)}
          disabled={submitting}
          required
        />
      </label>

      <label className="checkout-field">
        {t.phone}
        <input
          type="tel"
          dir="ltr"
          value={phone}
          onChange={(e) => setPhone(e.target.value)}
          disabled={submitting}
          required
        />
      </label>

      <label className="checkout-field">
        {t.address}
        <input
          type="text"
          value={deliveryAddress}
          onChange={(e) => setDeliveryAddress(e.target.value)}
          disabled={submitting}
          required
        />
      </label>

      <div className="checkout-payment-row">
        <span>{t.payment}</span>
        <span className="checkout-payment-value">{t.cod}</span>
      </div>

      {error && <div className="checkout-error">{error}</div>}

      <button type="submit" className="checkout-submit-button" disabled={submitting || !isFormFilled(fields)}>
        {submitting ? t.submitting : t.submit}
      </button>
    </form>
  );
}
