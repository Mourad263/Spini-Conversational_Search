import { useEffect, useRef, useState } from "react";
import { getSessionId, newSessionId } from "./session.js";
import { sendMessage, mutateCartItem } from "./api.js";
import MessageBubble from "./components/MessageBubble.jsx";
import ChatInput from "./components/ChatInput.jsx";
import CartPanel from "./components/CartPanel.jsx";

// Exact backend greeting (src/response/messages.py's GREETING_EN) --
// hardcoded here deliberately: it is shown with zero API/LLM calls.
const GREETING_EN = "Hello! I'm Spini. How can I help you?";
const ERROR_TEXT = "Sorry, I'm having trouble reaching the server. Please try again in a moment.";

// FINAL FRONTEND RESET task: the server includes `cart` only on cart-
// relevant turns (SessionMessageResponse.cart is null otherwise -- the
// cart hasn't changed then), so this default is only ever the actual
// starting state (a fresh session's empty cart), never a guess.
const EMPTY_CART = { lines: [], total: 0 };

function makeGreetingTurn() {
  return { id: crypto.randomUUID(), role: "assistant", text: GREETING_EN, products: [], recommendations: [] };
}

export default function App() {
  const [sessionId, setSessionId] = useState(() => getSessionId());
  const [messages, setMessages] = useState(() => [makeGreetingTurn()]);
  const [input, setInput] = useState("");
  const [pending, setPending] = useState(false);
  const [cart, setCart] = useState(EMPTY_CART);
  // product_ids with an in-flight button-driven cart mutation --
  // ProductCard/CartPanel disable that ONE product's +/- controls while its
  // own id is in here, so a double-click can never race two deltas for the
  // same line.
  const [pendingCartIds, setPendingCartIds] = useState(() => new Set());
  const [cartError, setCartError] = useState(null);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, pending]);

  // Applies any server-side checkout invalidation the SAME way the chat
  // path already does -- a button-driven mutation invalidates an open
  // draft exactly like a chat-driven one (session_activity.py's cart
  // methods call clear_checkout() either way), so this is the ONE place
  // both paths converge on dropping a stale checkoutForm.
  function applyCheckoutInvalidation(invalidated) {
    if (!invalidated) return;
    setMessages((prev) => prev.map((t) => (t.checkoutForm ? { ...t, checkoutForm: null } : t)));
  }

  // The ONE deterministic, 0-LLM cart mutation path -- ProductCard's
  // stepper and the cart dropdown's +/- both call this directly, NEVER
  // sendMessage()/the conversational endpoint. Replaces React cart state
  // with the server's own authoritative snapshot; never computes price/
  // total/quantity itself.
  async function handleCartDelta(productId, delta) {
    setPendingCartIds((prev) => new Set(prev).add(productId));
    setCartError(null);
    try {
      const result = await mutateCartItem(sessionId, productId, delta);
      setCart(result.cart); // authoritative snapshot replaces state outright
      applyCheckoutInvalidation(result.checkout_invalidated);
    } catch (err) {
      // Keep the previous authoritative cart state on error -- never fake
      // success, never guess a new total.
      setCartError(err.message || "Could not update your cart.");
    } finally {
      setPendingCartIds((prev) => {
        const next = new Set(prev);
        next.delete(productId);
        return next;
      });
    }
  }

  // `textOverride` lets a non-typed action (the cart dropdown's "Proceed to
  // checkout") send a message through the SAME existing conversational
  // flow as a typed Send -- never a second checkout endpoint/code path.
  async function handleSend(textOverride) {
    const text = (textOverride ?? input).trim();
    if (!text || pending) return; // guards against duplicate submission while loading

    const userTurn = { id: crypto.randomUUID(), role: "user", text };
    setMessages((prev) => [...prev, userTurn]);
    if (textOverride === undefined) setInput("");
    setPending(true);

    try {
      // Exactly one backend call per Send -- assistant_message/products/
      // recommendations all arrive together in this one response.
      const data = await sendMessage(sessionId, text);
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          text: data.assistant_message || "",
          products: data.products || [],
          recommendations: data.recommendations || [],
          checkoutForm: data.checkout_form || null,
        },
      ]);
      // null on an unrelated turn (the cart didn't change) -- keep showing
      // the last known snapshot rather than blanking it.
      if (data.cart) setCart(data.cart);
      // A cart mutation always invalidates any open checkout draft
      // server-side -- drop any checkoutForm still showing from an earlier
      // turn so it can never be submitted with stale products/quantities.
      if (data.checkout_invalidated) {
        setMessages((prev) => prev.map((t) => (t.checkoutForm ? { ...t, checkoutForm: null } : t)));
      }
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          text: ERROR_TEXT,
          isError: true,
          products: [],
          recommendations: [],
        },
      ]);
    } finally {
      setPending(false);
    }
  }

  function handleCheckoutSuccess(turnId, result) {
    // Removes the now-stale checkout form from its original turn (so it can
    // never be submitted again) and appends the confirmation as a NEW
    // Spini message -- the same shape a normal turn uses (text + order).
    setMessages((prev) => [
      ...prev.map((t) => (t.id === turnId ? { ...t, checkoutForm: null } : t)),
      {
        id: crypto.randomUUID(),
        role: "assistant",
        text: result.assistant_message || "",
        products: [],
        recommendations: [],
        order: result.order || null,
      },
    ]);
    // Purchased cart lines (if any) are already removed server-side by
    // this point.
    if (result.cart) setCart(result.cart);
  }

  function handleNewChat() {
    // A new session_id is enough -- backend's own 45-minute inactivity
    // expiration already reclaims the old session; nothing to delete here.
    const id = newSessionId();
    setSessionId(id);
    setMessages([makeGreetingTurn()]);
    setInput("");
    setCart(EMPTY_CART);
    setPendingCartIds(new Set());
    setCartError(null);
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <span className="app-title">Conversational Shopping Assistant</span>
        <div className="app-header-actions">
          {/* Keyed by sessionId so New Chat also resets CartPanel's own
              `open` dropdown state -- that state lives inside CartPanel, so
              remounting it on a new session is the existing-React way to
              reset it without a second reset implementation. */}
          <CartPanel
            key={sessionId}
            cart={cart}
            pendingCartIds={pendingCartIds}
            onCartDelta={handleCartDelta}
            onCheckout={() => handleSend("checkout")}
          />
          <button className="new-chat-button" onClick={handleNewChat}>
            New Chat
          </button>
        </div>
      </header>

      <main className="chat-window">
        {cartError && (
          <div className="cart-error-banner" role="alert">
            {cartError}
          </div>
        )}
        {messages.map((turn) => (
          <MessageBubble
            key={turn.id}
            turn={turn}
            sessionId={sessionId}
            onCheckoutSuccess={handleCheckoutSuccess}
            cart={cart}
            pendingCartIds={pendingCartIds}
            onCartDelta={handleCartDelta}
          />
        ))}
        {pending && (
          <div className="bubble-row bubble-row-assistant">
            <div className="bubble bubble-assistant bubble-loading" aria-label="Assistant is typing">
              <span className="typing-dot" />
              <span className="typing-dot" />
              <span className="typing-dot" />
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </main>

      <footer className="chat-input-area">
        <ChatInput value={input} onChange={setInput} onSend={handleSend} disabled={pending} />
      </footer>
    </div>
  );
}
