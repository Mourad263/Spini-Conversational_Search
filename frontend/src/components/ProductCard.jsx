import { isArabicText } from "../arabic.js";

function formatBrand(brand) {
  if (!brand) return null;
  return brand
    .split(/[_\-\s]+/)
    .filter(Boolean)
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

// FINAL FRONTEND RESET task: original card, unchanged, plus the minimum
// addition needed to use the deterministic cart endpoint -- `cart` is the
// server's own authoritative snapshot (App.jsx's `cart` state); this card
// never keeps its own quantity state, it derives what to show purely from
// `cart.lines` by product_id on every render. `onCartDelta` is the ONE
// deterministic, 0-LLM cart-mutation path (POST /session/{id}/cart/item);
// a button here NEVER calls sendMessage()/the conversational endpoint.
// `cart`/`pendingCartIds`/`onCartDelta` are all optional -- CheckoutForm
// renders this same card without them and simply gets no cart controls.
export default function ProductCard({ product, cart, pendingCartIds, onCartDelta }) {
  const brand = formatBrand(product.brand_normalized);
  const hasSpecial =
    product.special_price != null &&
    product.price != null &&
    product.special_price < product.price;

  const cartLine = cart?.lines?.find((l) => l.id === product.id);
  const quantity = cartLine?.quantity || 0;
  const isPending = pendingCartIds?.has(product.id) || false;
  const canMutate = typeof onCartDelta === "function" && product.price != null;

  function handleDelta(delta) {
    if (!canMutate || isPending) return;
    onCartDelta(product.id, delta);
  }

  return (
    <div className="product-card">
      <div className="product-name">{product.name_en}</div>
      {product.name_ar && (
        <div className="product-name-ar" dir={isArabicText(product.name_ar) ? "rtl" : "ltr"}>
          {product.name_ar}
        </div>
      )}
      {brand && <div className="product-brand">{brand}</div>}
      <div className="product-price-row">
        {product.price != null && (
          <span className={hasSpecial ? "product-price product-price-strike" : "product-price"}>
            {product.price.toFixed(2)} EGP
          </span>
        )}
        {hasSpecial && <span className="product-special-price">{product.special_price.toFixed(2)} EGP</span>}
      </div>
      {canMutate && (
        quantity === 0 ? (
          <div className="stepper stepper-add-only">
            <button
              type="button"
              className="stepper-add"
              onClick={() => handleDelta(1)}
              disabled={isPending}
              aria-label="Add to cart"
            >
              Add to cart
            </button>
          </div>
        ) : (
          <div className="stepper">
            <button
              type="button"
              className="stepper-btn stepper-remove"
              onClick={() => handleDelta(-1)}
              disabled={isPending}
              aria-label="Remove one"
            >
              −
            </button>
            <span className="stepper-qty">{quantity}</span>
            <button
              type="button"
              className="stepper-btn"
              onClick={() => handleDelta(1)}
              disabled={isPending}
              aria-label="Add one"
            >
              +
            </button>
          </div>
        )
      )}
    </div>
  );
}
