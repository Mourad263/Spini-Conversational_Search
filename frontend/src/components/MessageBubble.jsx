import { isArabicText } from "../arabic.js";
import ProductSection from "./ProductSection.jsx";
import CheckoutForm from "./CheckoutForm.jsx";
import OrderConfirmation from "./OrderConfirmation.jsx";

export default function MessageBubble({ turn, sessionId, onCheckoutSuccess, cart, pendingCartIds, onCartDelta }) {
  const arabic = isArabicText(turn.text);
  const bubbleClass = `bubble bubble-${turn.role}${turn.isError ? " bubble-error" : ""}`;
  const recLabel = arabic ? "مقترح لك" : "Recommended for you";

  return (
    <div className={`bubble-row bubble-row-${turn.role}`}>
      <div className={bubbleClass} dir={arabic ? "rtl" : "ltr"}>
        {turn.text}
      </div>
      {turn.role === "assistant" && (
        <>
          <ProductSection products={turn.products} cart={cart} pendingCartIds={pendingCartIds} onCartDelta={onCartDelta} />
          <ProductSection title={recLabel} products={turn.recommendations} cart={cart} pendingCartIds={pendingCartIds} onCartDelta={onCartDelta} />
          {turn.checkoutForm && (
            <CheckoutForm
              form={turn.checkoutForm}
              sessionId={sessionId}
              arabic={arabic}
              onSuccess={(result) => onCheckoutSuccess(turn.id, result)}
            />
          )}
          {turn.order && <OrderConfirmation order={turn.order} arabic={arabic} />}
        </>
      )}
    </div>
  );
}
