import ProductCard from "./ProductCard.jsx";

// Used for both plain search `products` (no title) and `recommendations`
// (labeled, e.g. "Recommended for you") -- never merged into one list.
// `cart`/`pendingCartIds`/`onCartDelta` just pass straight through so each
// ProductCard can derive its OWN displayed quantity from the authoritative
// cart snapshot and drive the deterministic cart endpoint directly.
export default function ProductSection({ title, products, cart, pendingCartIds, onCartDelta }) {
  if (!products || products.length === 0) return null;
  return (
    <div className="product-section">
      {title && <div className="product-section-title">{title}</div>}
      <div className="product-grid">
        {products.map((p) => (
          <ProductCard
            key={p.id}
            product={p}
            cart={cart}
            pendingCartIds={pendingCartIds}
            onCartDelta={onCartDelta}
          />
        ))}
      </div>
    </div>
  );
}
