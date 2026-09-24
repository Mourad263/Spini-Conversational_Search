import { useEffect, useRef } from "react";

export default function ChatInput({ value, onChange, onSend, disabled }) {
  const inputRef = useRef(null);

  // Sending disables the textarea while the request is in flight, which the
  // browser forces a blur on -- refocus the moment it re-enables (the
  // response arrived) so Enter/Send/Enter/Send works without reclicking.
  useEffect(() => {
    if (!disabled) {
      inputRef.current?.focus();
    }
  }, [disabled]);

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      onSend();
    }
    // Shift+Enter: default textarea behavior (newline) is left alone.
  }

  return (
    <div className="chat-input-row">
      <textarea
        ref={inputRef}
        className="chat-input"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Type your message..."
        rows={1}
        disabled={disabled}
      />
      <button className="send-button" onClick={() => onSend()} disabled={disabled || !value.trim()}>
        Send
      </button>
    </div>
  );
}
