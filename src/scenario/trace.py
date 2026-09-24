"""Sprint 6 observability task: one lightweight, server-side-only trace per
conversational message (src/scenario/orchestrator.py's handle_message()).
Observation only -- never influences routing/behavior, never reaches the
customer-facing API response (src/api/schemas.py's SessionMessageResponse
has no trace fields). No OpenTelemetry/Prometheus/DB -- one dataclass, one
structured log line via the existing `logging` module.

Instructor-demo observability task: extends the SAME dataclass (never a
second tracing system) with purely-observational fields describing what
happened during the request -- session state before/after, the
deterministic Action interpretation used (never LLM chain-of-thought, just
the already-structured intent/cart_action/selection_* fields), which
product-resolution authority/pool was consulted, and a response summary.
Every field here is set from data that ALREADY exists elsewhere (activity
snapshots, the Action object, _resolve_product_selection()'s own return
value, the final SessionMessageResponse) -- setting them is always a
read, never a decision, so tracing can never change behavior. format_human_
trace() renders this into the terminal block a human reads; nothing here
is sent to the customer-facing API response or influences routing.

SECURITY: never populate a field with a customer PII value (name/phone/
address) -- see the checkout_submission_* fields below, which are booleans
computed by the caller (src/scenario/checkout_submission.py), never the
raw PII itself.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import asdict, dataclass, field


def spini_trace_enabled() -> bool:
    """SPINI_TRACE=1 (any other value, or unset, leaves normal logging
    unchanged) -- checked fresh on every call so a test/subprocess can
    toggle it via monkeypatch/env without reimporting this module."""
    return os.environ.get("SPINI_TRACE") == "1"


@dataclass
class RequestTrace:
    session_id: str
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    route: str = "unknown"  # the actual branch handle_message took, e.g.
    # shopping_search/recommendation/memory_recall/off_topic/reset/blocked/error
    total_latency_ms: float | None = None

    llm_calls: int = 0  # incremented only when understand_fn is actually invoked
    llm_input_tokens: int | None = None  # real usage.prompt_tokens, never estimated
    llm_output_tokens: int | None = None  # real usage.completion_tokens
    llm_total_tokens: int | None = None  # real usage.total_tokens

    validation_calls: int = 0  # validate() actually called
    search_calls: int = 0  # search_fn actually called (direct search + any
    # embedded in recommendation evidence-gathering)
    recommendation_calls: int = 0  # RecommendationService.recommend() actually called
    embedding_calls: int | None = None  # real embed() count for this request

    # --- instructor-demo observability task: human-readable trace fields.
    # `message_text` is the raw user turn (never LLM reasoning) -- for the
    # structured checkout-submission form POST (no free-text message),
    # src/scenario/checkout_submission.py sets a fixed placeholder instead.
    message_text: str | None = None

    # session state BEFORE this request was processed
    shown_products_before: int = 0
    recent_pages_before: int = 0
    cart_items_before: int = 0
    pending_action_before: str | None = None
    checkout_draft_before: bool = False
    phase_before: str | None = None  # deterministic conversation-state task:
    # src/scenario/llm_context.py's derive_conversation_phase() -- a DERIVED
    # read of already-authoritative fields above, never a second state source

    # deterministic conversation-state/compact-memory task: whether a
    # compact, read-only session-memory block was actually sent to
    # understand() this turn (see llm_context.py's build_llm_session_
    # context()) -- a presence flag/metadata only, never the block's own
    # text (that would just be the LLM prompt again -- see this module's
    # own docstring on never logging the raw prompt).
    session_context_sent: bool = False

    # the deterministic Action actually used to decide this turn -- either
    # the real understand() result, or (A2.5c) a purely raw-text-derived
    # synthetic Action. Never set from LLM free-text/reasoning, only from
    # the already-structured, already-validated Action fields.
    action_intent: str | None = None
    action_cart_action: str | None = None
    action_checkout_request: bool | None = None
    action_selection_ordinals: list[int] | None = None
    action_selection_text: str | None = None

    # the ONE product-resolution authority actually consulted, if any (see
    # orchestrator.py's _trace_resolve(), the one wrapper around the one
    # resolver, _resolve_product_selection())
    resolution_authority: str | None = None
    resolution_product_ids: list[int] | None = None
    resolution_reason: str | None = None

    # session state AFTER this request was processed
    shown_products_after: int = 0
    cart_items_after: int = 0
    pending_action_after: str | None = None
    checkout_draft_after: bool = False
    checkout_items_after: int = 0
    phase_after: str | None = None  # derive_conversation_phase() on the "after" snapshot

    # response summary -- counts/flags only, never response text/PII
    response_status: str | None = None
    response_product_count: int = 0
    response_cart_items: int = 0
    response_checkout_form: bool = False

    # checkout-submission-only fields, set by src/scenario/
    # checkout_submission.py -- deliberately booleans/counts, NEVER the
    # actual customer_name/phone/delivery_address (see this module's own
    # docstring SECURITY note).
    checkout_submission_items: int | None = None
    checkout_submission_name_present: bool | None = None
    checkout_submission_phone_valid: bool | None = None
    checkout_submission_address_present: bool | None = None

    def to_log_dict(self) -> dict:
        return asdict(self)

    @property
    def llm_called(self) -> bool:
        return self.llm_calls > 0

    @property
    def fast_path(self) -> bool:
        """True whenever this request was resolved entirely without an
        understand_fn() call -- a social/blocked/pending-action fast path,
        or a deterministic A2.5c synthetic-Action resolution."""
        return self.llm_calls == 0


def _fmt(value: object) -> str:
    return "n/a" if value is None else str(value)


def format_human_trace(trace: RequestTrace) -> str:
    """Renders the exact instructor-demo terminal block. Execution-trace
    only -- every value here already exists on `trace`, itself populated
    only from already-computed state/Action/response data (see this
    module's docstring); no LLM chain-of-thought is ever included."""
    bar = "─" * 64
    lines = [
        bar,
        f"[SPINI TRACE] request_id={trace.request_id} session={trace.session_id}",
        "",
        "MESSAGE",
        f'  text="{trace.message_text or ""}"',
        "",
        "STATE BEFORE",
        f"  phase={_fmt(trace.phase_before)}",
        f"  shown_products={trace.shown_products_before}",
        f"  recent_pages={trace.recent_pages_before}",
        f"  cart_items={trace.cart_items_before}",
        f"  pending_action={_fmt(trace.pending_action_before)}",
        f"  checkout_draft={trace.checkout_draft_before}",
        "",
        "INTERPRETATION",
        f"  fast_path={trace.fast_path}",
        f"  llm_called={trace.llm_called}",
        f"  session_context_sent={trace.session_context_sent}",
        f"  intent={_fmt(trace.action_intent)}",
        f"  cart_action={_fmt(trace.action_cart_action)}",
        f"  checkout_request={_fmt(trace.action_checkout_request)}",
        f"  selection_ordinals={_fmt(trace.action_selection_ordinals)}",
        f"  selection_text={_fmt(trace.action_selection_text)}",
        "",
        "RESOLUTION",
        f"  authority={_fmt(trace.resolution_authority)}",
        f"  product_ids={_fmt(trace.resolution_product_ids)}",
        f"  reason={_fmt(trace.resolution_reason)}",
        "",
        "ROUTE",
        f"  {trace.route}",
        "",
        "STATE MUTATION",
        f"  cart: {trace.cart_items_before} -> {trace.cart_items_after}",
        f"  pending_action: {_fmt(trace.pending_action_before)} -> {_fmt(trace.pending_action_after)}",
        f"  checkout_draft: {trace.checkout_draft_before} -> {trace.checkout_draft_after}",
        "",
        "STATE AFTER",
        f"  phase={_fmt(trace.phase_after)}",
        f"  shown_products={trace.shown_products_after}",
        f"  cart_items={trace.cart_items_after}",
        f"  pending_action={_fmt(trace.pending_action_after)}",
        f"  checkout_draft={trace.checkout_draft_after}",
        f"  checkout_items={trace.checkout_items_after}",
        "",
        "RESPONSE",
        f"  status={_fmt(trace.response_status)}",
        f"  product_count={trace.response_product_count}",
        f"  cart_items={trace.response_cart_items}",
        f"  checkout_form={trace.response_checkout_form}",
        bar,
    ]
    if trace.checkout_submission_items is not None:
        lines.insert(-1, "CHECKOUT SUBMISSION")
        lines.insert(-1, f"  items={trace.checkout_submission_items}")
        lines.insert(-1, f"  name_present={trace.checkout_submission_name_present}")
        lines.insert(-1, f"  phone_valid={trace.checkout_submission_phone_valid}")
        lines.insert(-1, f"  address_present={trace.checkout_submission_address_present}")
        lines.insert(-1, "")
    return "\n".join(lines)


def print_human_trace(trace: RequestTrace) -> None:
    """Prints format_human_trace()'s block to stdout. Wrapped so a console
    encoding issue (e.g. a non-UTF-8 Windows terminal choking on the
    box-drawing border) can never raise out of here -- tracing is
    observation only and must never crash/alter the real request."""
    try:
        print(format_human_trace(trace))
    except UnicodeEncodeError:
        print(format_human_trace(trace).encode("ascii", errors="replace").decode("ascii"))
