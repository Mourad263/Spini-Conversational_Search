import { useState } from "react";
import { isArabicText } from "../arabic.js";

// FINAL FRONTEND RESET task: the original frontend had no cart UI at all
// (cart/checkout didn't exist yet) -- this is the smallest addition that
// makes the existing deterministic cart endpoint reachable: one header
// button, and a small dropdown that is CLOSED by default and has zero
// layout footprint until clicked (display:none, not just hidden offscreen).
// No bottom sheet, no permanent panel, no overlay/backdrop.
export default function CartPanel({ cart, pendingCartIds, onCartDelta, onCheckout }) {
  const [open, setOpen] = useState(false);
  const lines = cart?.lines || [];
  const lineCount = lines.reduce((sum, l) => sum + l.quantity, 0);
  const total = cart?.total || 0;

  function handleCheckout() {
    setOpen(false);
    onCheckout();
  }

  return (
    <div>
      <button
        type="button"
        className="cart-button"
        onClick={() => setOpen((o) => !o)}
        aria-haspopup="true"
        aria-expanded={open}
      >
        <svg
          className="cart-icon"
          viewBox="0 0 24 24"
          width="18"
          height="18"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden="true"
        >
          <circle cx="9" cy="21" r="1" />
          <circle cx="20" cy="21" r="1" />
          <path d="M1 1h4l2.68 13.39a2 2 0 0 0 2 1.61h9.72a2 2 0 0 0 2-1.61L23 6H6" />
        </svg>
        <span className="cart-badge">{lineCount}</span>
      </button>

      <div className={`cart-dropdown${open ? " open" : ""}`} role="dialog" aria-label="Cart">
        <div className="cart-dropdown-title">Your cart</div>
        {lines.length === 0 ? (
          <div className="cart-dropdown-empty">Your cart is empty.</div>
        ) : (
          <>
            {lines.map((line) => {
              const isPending = pendingCartIds?.has(line.id) || false;
              return (
                <div className="cart-dropdown-row" key={line.id}>
                  <div className="cart-dropdown-name">
                    <div>{line.name_en}</div>
                    {line.name_ar && (
                      <div className="ar" dir={isArabicText(line.name_ar) ? "rtl" : "ltr"}>
                        {line.name_ar}
                      </div>
                    )}
                  </div>
                  <div className="stepper cart-dropdown-stepper">
                    <button
                      type="button"
                      className="stepper-btn stepper-remove"
                      onClick={() => onCartDelta(line.id, -1)}
                      disabled={isPending}
                      aria-label="Remove one"
                    >
                      −
                    </button>
                    <span className="stepper-qty">{line.quantity}</span>
                    <button
                      type="button"
                      className="stepper-btn"
                      onClick={() => onCartDelta(line.id, 1)}
                      disabled={isPending}
                      aria-label="Add one"
                    >
                      +
                    </button>
                  </div>
                  <div className="cart-dropdown-price">{line.line_total.toFixed(2)}</div>
                </div>
              );
            })}
            <div className="cart-dropdown-total">
              <span>Total</span>
              <span>{total.toFixed(2)} EGP</span>
            </div>
            <button type="button" className="cart-dropdown-checkout" onClick={handleCheckout}>
              Proceed to checkout
            </button>
          </>
        )}
      </div>
    </div>
  );
}
