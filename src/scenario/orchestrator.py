"""Sprint 4 Part 3: the thin conversational orchestration path that wires
together every already-built component into one real turn pipeline:

    pre-LLM guards -> understand() -> deterministic Action normalization
    -> validate()/StateManager (or the deterministic recommendation
    policy) -> SearchAdapter -> real product rows only -> deterministic
    response wording.

Deliberately NOT a new search/state abstraction -- StateManager,
validate(), parse_keyword_query(), and SearchAdapter are used exactly as
they already exist (src/state_manager, src/validation, src/search_engine,
src/search_adapter); this module only decides WHICH of them to call, in
WHICH order, for a given turn, per the Sprint 4 design/audit passes.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logger = logging.getLogger(__name__)
# Sprint 6 observability task: under a normal `uvicorn ... ` startup nothing
# has ever called logging.basicConfig()/configured the root logger, so this
# logger's effective level is the default WARNING and its request_trace INFO
# lines are silently dropped -- Uvicorn's own dictConfig only ever touches
# its own "uvicorn"/"uvicorn.error"/"uvicorn.access" loggers, never root or
# this one. Attaching a small dedicated handler directly here (rather than
# calling logging.basicConfig(), which would configure the ROOT logger and
# risk duplicating Uvicorn's own console output) makes traces visible
# without touching Uvicorn's logging setup at all. Guarded by `if not
# logger.handlers` so re-importing this module within the same process
# (e.g. multiple TestClient/app instances) never accumulates duplicate
# handlers/duplicate log lines -- a `--reload` code change instead restarts
# the whole worker process, which naturally starts with zero handlers again.
if not logger.handlers:
    _trace_handler = logging.StreamHandler()
    _trace_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_trace_handler)
logger.setLevel(logging.INFO)

from src.api.schemas import SessionMessageResponse  # noqa: E402
from src.recommendation.service import get_recommendation_service  # noqa: E402
from src.response import messages  # noqa: E402
from src.scenario.decision import (  # noqa: E402
    derive_recommendation_search,
    has_searchable_target,
    normalize_recommendation_intent,
    normalize_target_facet,
)
from src.scenario.guard import check_rapid_repeat, has_reached_turn_cap  # noqa: E402
from src.scenario.cart_flow import (  # noqa: E402
    _cart_and_recent_pool, _raw_message_mentions_cart, _to_cart_out, handle_cart_action,
    resolve_deterministic_cart_fast_path,
)
from src.scenario.checkout_flow import _checkout_from_lines, _raw_message_supports_checkout_all  # noqa: E402
from src.scenario.llm_context import (  # noqa: E402
    build_llm_session_context, derive_conversation_phase, summarize_turn_outcome,
)
from src.scenario.pending_actions import (  # noqa: E402
    resolve_pending_cart_add, resolve_pending_cart_checkout, resolve_pending_multi_add_to_cart,
    resolve_pending_recommendation_offer,
)
from src.scenario.product_selection import _selection_failure_message, _trace_resolve, _trace_set_action  # noqa: E402
from src.scenario.response_builder import _response, _to_displayed_products  # noqa: E402
from src.scenario.session_activity import CartLine, SessionActivity, SessionActivityStore  # noqa: E402
from src.scenario.social import GREETING, HOW_ARE_YOU, NAME_QUESTION, classify_social  # noqa: E402
from src.scenario.trace import RequestTrace, print_human_trace, spini_trace_enabled  # noqa: E402
from src.search_adapter.adapter import FilterSet, SearchAdapter, SearchResult  # noqa: E402
from src.search_engine.embeddings import pop_embedding_trace, reset_embedding_trace  # noqa: E402
from src.search_engine.index import get_index  # noqa: E402
from src.search_engine.keyword_baseline import _corpus_cooccurrence_ids  # noqa: E402
from src.search_engine.ranking import correct_unmatched_terms  # noqa: E402
from src.search_engine.tokenizer import tokenize  # noqa: E402
from src.state_manager.state_manager import StateManager  # noqa: E402
from src.understanding.llm import (  # noqa: E402
    pop_usage_trace, reset_usage_trace, set_selectable_products_shown, set_session_context, understand,
)
from src.understanding.schema import Action, Intent  # noqa: E402
from src.validation.validate import validate  # noqa: E402

_STATE_CHANGING_INTENTS = (
    Intent.SEARCH, Intent.ADD_FILTER, Intent.MODIFY_FILTER, Intent.REMOVE_FILTER,
)


def _real_search(filters: FilterSet, limit: int, offset: int) -> SearchResult:
    return SearchAdapter().search(filters, limit=limit, offset=offset)


def _likely_out_of_catalog(searched_filters: FilterSet, result: SearchResult) -> bool:
    """Honest, conservative signal (Sprint 4 design pass) -- NEVER
    "proven". True only when ALL of: a real search ran and found zero
    results, the searched free text is non-empty, no canonical
    category/brand was resolved (a resolved category/brand means the
    concept IS in the catalog -- this exact combination just returned
    nothing, a filter_conflict, not an absent concept), the literal query
    tokens occur nowhere in the entire corpus (checked via the EXISTING
    SearchIndex vocabulary, index.postings -- no new DB query, no new
    infrastructure), AND the SAME generic, corpus-backed spelling
    correction SearchAdapter/rank() already use internally
    (ranking.py's correct_unmatched_terms) finds no safe real-corpus
    replacement either.

    Audit fix (manual acceptance test, real query "لانشن تحت 50"): a
    correctable typo for an in-catalog term (لانشن -> لانشون, 90 real
    luncheon products, corpus-confirmed) was being labeled
    likely_out_of_catalog=True purely because the ORIGINAL misspelled
    token has zero literal overlap -- this function never consulted
    correct_unmatched_terms() before concluding that. A correction found
    by that EXISTING mechanism is itself positive catalog evidence (the
    corrected term is real, corpus-frequent vocabulary), even when the
    turn's own filters (e.g. an explicit low price ceiling) still
    legitimately return zero matching rows -- that is a filter_conflict/
    genuine zero-result case, not evidence the concept is absent. A
    severe/uncorrectable misspelling can still false-positive here --
    disclosed, not solved."""
    if not result.zero_result:
        return False
    if not searched_filters.query_text:
        return False
    if searched_filters.category is not None or searched_filters.brand is not None:
        return False
    index = get_index()
    tokens = tokenize(searched_filters.query_text, index.base_freq)
    if not tokens:
        return False
    if index.candidates_for_tokens(tokens):
        return False
    return correct_unmatched_terms(index, searched_filters.query_text) is None


def _lacks_catalog_evidence(resolved: FilterSet) -> bool:
    """Sprint 6 out-of-catalog grounding task: deterministic, PRE-search
    gate -- True only when THIS TURN's own resolution (never the
    post-merge canonical state) gives no reason to believe the requested
    concept is something this catalog sells at all. Corpus-vocabulary
    evidence only, via the SAME in-memory SearchIndex used elsewhere here
    -- no DB query, no embeddings, no vector call.

    A resolved canonical category or brand is authoritative and always
    passes: validate()'s category_hint/brand_text grounding is itself
    typo/dialect-tolerant (it is a semantic classification, not an exact
    spelling match), so this alone already covers the overwhelming
    majority of real, uncommon, Arabic, code-switched, or misspelled
    catalog products -- see resolve_category_hint/resolve_brand_text. An
    empty query_text also passes: nothing to judge.

    Only for a PURE free-text query (no category/brand resolved) does this
    look at query_text itself, and ONLY when it has 2+ words: it then
    requires the words to actually CO-OCCUR together in at least one real
    product (src/search_engine/keyword_baseline.py's
    _corpus_cooccurrence_ids -- the SAME evidence that module's own fuzzy-
    category/brand matching relies on, reused here rather than building a
    second vocabulary system). This is deliberately stronger than each
    word independently having SOME corpus presence: real regression
    measured directly on "laptop charger" -- "laptop" and "charger" each
    independently occur in a handful of unrelated products, but ZERO real
    products contain both words together, so independent-presence alone
    wrongly let it through. "كوتش عربية" (car tires) fails the same way:
    "كوتش" occurs in zero real products at all, so it can't co-occur with
    anything either. Genuine multi-word products ("oat milk", "chicken
    breast") have real products containing both words together and pass
    normally.

    A SINGLE-word query is deliberately exempt from this stricter rule
    and always passes (returns False here) -- found live, via the
    project's own existing regression suite
    (tests/integration/test_typo_tolerance_audit.py's direct-vs-
    conversational parity tests): a lone word with zero literal corpus
    presence is exactly the shape of a genuine, single-word typo this
    catalog's OWN calibrated typo/vector-rescue machinery (ranking.py's
    correct_unmatched_terms, vector_search.py's distance-gated fallback --
    e.g. "milq"->milk, "لانشن"->لانشون/luncheon, both real, extensively
    measured rescues) is specifically built to recover; this gate must
    not pre-empt that with a cruder, unweighted rule. The 2+-word
    requirement is what actually distinguishes the two cases: a whole
    CONCEPT (multiple words) with no product ever combining them is a much
    stronger absence signal than one isolated word standing alone."""
    if resolved.category is not None or resolved.brand is not None:
        return False
    if not resolved.query_text:
        return False
    words = resolved.query_text.split()
    if len(words) < 2:
        return False
    return not _corpus_cooccurrence_ids(words)


def _build_recommendations(
    action: Action, filters: FilterSet, search_fn: Callable[[FilterSet, int, int], SearchResult], limit: int,
    trace: RequestTrace | None = None, *, known_shown_ids: list[int] | None = None,
) -> list[dict]:
    """Read-only: never calls StateManager.apply(), never mutates `filters`.
    `known_shown_ids`, when given, is used as-is instead of running a fresh
    search under `filters` -- callers pass this only when they already have
    an ACCURATE id list for `filters` on hand (see the E branch below), so a
    real search under `filters` isn't repeated just to rediscover it. []
    whenever recommendation_request is false, there is no searchable target,
    or the evidence (fresh search or known_shown_ids) is empty -- never a
    fallback to an unfiltered/looser result. `trace` (Sprint 6 observability
    task) is purely observational -- when given, its recommendation_calls is
    incremented only at the exact line RecommendationService.recommend()
    actually runs; search_calls is left to `search_fn` itself (orchestrator
    passes an already-counting wrapper)."""
    if not action.recommendation_request or not has_searchable_target(filters):
        return []
    if known_shown_ids is None:
        base_result = search_fn(filters, limit, 0)
        if not base_result.products:
            return []
        known_shown_ids = [p["id"] for p in base_result.products]
    if not known_shown_ids:
        return []
    if trace is not None:
        trace.recommendation_calls += 1
    return get_recommendation_service().recommend(filters.query_text, known_shown_ids, filters, limit=limit)


def _search_message_parts(
    action: Action, searched_filters: FilterSet, result: SearchResult, activity: SessionActivity,
    *, relaxation_note: bool = False,
) -> tuple[list[str], bool]:
    parts: list[str] = []
    if action.customer_service_tone:
        parts.append(messages.customer_service_acknowledgment())
    if relaxation_note:
        parts.append(messages.recommendation_band_note())
    if searched_filters.unresolved_exclusion:
        parts.append(messages.unresolved_exclusion_note(searched_filters.unresolved_exclusion))
    likely_ooc = _likely_out_of_catalog(searched_filters, result)
    if result.zero_result:
        parts.append(messages.zero_result(likely_ooc))
    if has_reached_turn_cap(activity):
        parts.append(messages.turn_cap_suggestion())
    return parts, likely_ooc


def _out_of_catalog_message_parts(action: Action, resolved: FilterSet, activity: SessionActivity) -> list[str]:
    """Sprint 6 out-of-catalog grounding task: wording for a request the
    PRE-search gate (_lacks_catalog_evidence) already rejected. Built
    directly, NOT via _search_message_parts -- that helper's own
    likely_out_of_catalog recomputation uses the older, looser ANY-token+
    spelling-correction heuristic (calibrated for post-hoc wording on a
    genuine zero-result SEARCH), which can disagree with this gate's
    stricter ALL-tokens rule on the exact cases the gate exists to catch
    (e.g. "كوتش عربية" -- see _lacks_catalog_evidence's docstring). The
    gate's decision is authoritative here; it is never re-derived."""
    parts: list[str] = []
    if action.customer_service_tone:
        parts.append(messages.customer_service_acknowledgment())
    if resolved.unresolved_exclusion:
        parts.append(messages.unresolved_exclusion_note(resolved.unresolved_exclusion))
    parts.append(messages.zero_result(True))
    if has_reached_turn_cap(activity):
        parts.append(messages.turn_cap_suggestion())
    return parts


def handle_message(
    session_id: str,
    message: str,
    *,
    state_manager: StateManager,
    activity_store: SessionActivityStore,
    understand_fn: Callable[..., Action] = understand,
    search_fn: Callable[[FilterSet, int, int], SearchResult] = _real_search,
    now: float | None = None,
    limit: int = 20,
    offset: int = 0,
) -> SessionMessageResponse:
    """Public entry point -- unchanged behavior/signature. Sprint 6
    observability task: owns exactly one RequestTrace per call, timing and
    logging it in `finally` so every return path in _handle_message_inner
    (including an unreachable-branch exception) still emits it. Tracing
    never influences the response -- see src/scenario/trace.py.

    Instructor-demo observability task: `response` is captured in this
    outer scope (not read inside _handle_message_inner, never influencing
    any branch there) purely so `finally` can take one read-only "after"
    state snapshot and a response summary -- an exception leaves `response`
    None, and the STATE AFTER/RESPONSE trace sections are simply skipped."""
    trace = RequestTrace(session_id=session_id)
    start = time.monotonic()
    reset_usage_trace()
    reset_embedding_trace()
    response: SessionMessageResponse | None = None
    # Deterministic conversation-state task: read BEFORE processing, purely
    # to detect (by comparison, after) whether THIS call appended a new
    # TurnRecord -- never used for anything else, never influences routing.
    turns_before = len(activity_store.get(session_id).turns)
    try:
        response = _handle_message_inner(
            session_id, message, state_manager=state_manager, activity_store=activity_store,
            understand_fn=understand_fn, search_fn=search_fn, now=now, limit=limit, offset=offset,
            trace=trace,
        )
        return response
    finally:
        trace.total_latency_ms = round((time.monotonic() - start) * 1000, 2)
        usage = pop_usage_trace()
        if usage is not None:
            trace.llm_input_tokens = usage.get("input_tokens")
            trace.llm_output_tokens = usage.get("output_tokens")
            trace.llm_total_tokens = usage.get("total_tokens")
        trace.embedding_calls = pop_embedding_trace()
        # Read-only "after" snapshot -- activity_store.get() with no `now`
        # never checks expiry or stamps last_active, so this extra read
        # cannot change session state or TTL behavior.
        after = activity_store.get(session_id)
        trace.shown_products_after = len(after.last_shown_products)
        trace.cart_items_after = len(after.cart)
        trace.pending_action_after = after.pending_action.kind if after.pending_action else None
        trace.checkout_draft_after = after.checkout_active
        trace.checkout_items_after = len(after.selected_products)
        trace.phase_after = derive_conversation_phase(after)
        if response is not None:
            trace.response_status = response.intent
            trace.response_product_count = len(response.products)
            trace.response_cart_items = len(response.cart.lines) if response.cart else 0
            trace.response_checkout_form = response.checkout_form is not None
            # Deterministic conversation-state/compact-memory task: a short,
            # fully deterministic summary of what THIS turn did (see
            # llm_context.py's summarize_turn_outcome()) -- becomes Layer 2
            # conversational memory for a LATER turn's compact LLM context.
            # Never set when no turn was actually appended this call (a
            # blocked/social-fast-path turn, or an exception).
            if len(after.turns) > turns_before:
                activity_store.set_last_turn_outcome(session_id, summarize_turn_outcome(trace, response))
        logger.info("request_trace %s", trace.to_log_dict())
        if spini_trace_enabled():
            print_human_trace(trace)


def _handle_message_inner(
    session_id: str,
    message: str,
    *,
    state_manager: StateManager,
    activity_store: SessionActivityStore,
    trace: RequestTrace,
    understand_fn: Callable[..., Action] = understand,
    search_fn: Callable[[FilterSet, int, int], SearchResult] = _real_search,
    now: float | None = None,
    limit: int = 20,
    offset: int = 0,
) -> SessionMessageResponse:
    """One conversational turn, exactly per the Sprint 4 Part 3 pipeline
    order. `message` is assumed already length-validated by the FastAPI
    request schema (SessionMessageRequest, max_length=500) -- that check
    happens before this function is ever called, so an oversized request
    never reaches understand(). `understand_fn`/`search_fn`/`now` are
    injectable purely for testing (no real OpenAI/DB calls needed in
    orchestration-logic tests); the real route always uses the defaults."""
    now = time.monotonic() if now is None else now

    def _counted_search(filters: FilterSet, limit_: int, offset_: int) -> SearchResult:
        # Sprint 6 observability task: the ONE place search_fn is actually
        # invoked from here on -- counts real execution only, never presence
        # in the pipeline (see _build_recommendations, which receives this
        # same wrapper instead of the raw search_fn).
        trace.search_calls += 1
        return search_fn(filters, limit_, offset_)

    # A1: lazy session-expiration check (Sprint 6 lifecycle task) -- no
    # background worker, just a check on this real access. SessionActivity
    # is the single last-activity clock for the whole session; when it says
    # this session id has been idle past SESSION_TTL_SECONDS, StateManager's
    # independent shopping FilterSet is dropped here (its own store has no
    # activity clock of its own) BEFORE activity_store.get() below lazily
    # wipes its own half -- an expired session must lose BOTH.
    if activity_store.is_expired(session_id, now):
        state_manager.clear(session_id)
    activity = activity_store.get(session_id, now=now)
    trace.message_text = message
    # Instructor-demo observability task: read-only "before" snapshot,
    # taken here (before A2.5's pop_pending_action() below consumes
    # pending_action, and before anything else this turn can mutate) so it
    # reflects real session state as this request STARTED.
    trace.shown_products_before = len(activity.last_shown_products)
    trace.recent_pages_before = len(activity.recent_shown_pages)
    trace.cart_items_before = len(activity.cart)
    trace.pending_action_before = activity.pending_action.kind if activity.pending_action else None
    trace.checkout_draft_before = activity.checkout_active
    trace.phase_before = derive_conversation_phase(activity)

    # Abandoned-session reclaim task: the check above only ever revisits
    # THIS session_id -- an old id that's simply never accessed again (a
    # "New Chat" moving on to a fresh one) never gets cleaned up on its
    # own, growing both stores forever. Piggybacked here (amortized, see
    # SessionActivityStore's _RECLAIM_CHECK_EVERY) rather than a
    # background thread/scheduler; clears BOTH stores for every session it
    # reclaims, same split as the expiry check just above.
    for abandoned_id in activity_store.reclaim_abandoned(now, exclude=session_id):
        state_manager.clear(abandoned_id)

    # A2: rapid-repeat guard. Blocked: no understand(), no history append,
    # no accepted turn_count increment, no StateManager mutation.
    if not check_rapid_repeat(activity, message, now=now):
        trace.route = "blocked"
        state = state_manager.get_state(session_id)
        return _response(session_id, messages.rapid_repeat_blocked(), "blocked", state,
                          user_message=message, blocked=True)

    # A2.5/A2.5b/A2.5c: resolution of a PendingAction left by a PRIOR turn
    # (a recommendation offer, a proposed multi-product cart add, or a
    # cart-checkout scope clarification) -- see src/scenario/
    # pending_actions.py for the full contract. pop_pending_action() clears
    # it unconditionally BEFORE any resolver runs, which is what guarantees
    # a stale pending action can never hijack a LATER, unrelated turn; each
    # resolver returns None when this message doesn't answer it, and
    # falling through here means the SAME message continues to the normal
    # pipeline (social fast path, then understand()) below.
    pending = activity_store.pop_pending_action(session_id)
    if pending is not None and pending.kind == "recommendation_offer":
        result = resolve_pending_recommendation_offer(
            session_id, message, pending, state_manager, activity_store, trace, limit,
        )
        if result is not None:
            return result
    if pending is not None and pending.kind == "multi_add_to_cart_pending":
        result = resolve_pending_multi_add_to_cart(
            session_id, message, activity, pending, state_manager, activity_store, trace,
        )
        if result is not None:
            return result
    if pending is not None and pending.kind == "cart_checkout_pending":
        result = resolve_pending_cart_checkout(
            session_id, message, activity, pending, state_manager, activity_store, trace,
        )
        if result is not None:
            return result
    # A2.5d (deterministic conversation-state task): pending cart-add
    # product-selection clarification -- see src/scenario/pending_actions.py's
    # resolve_pending_cart_add(). The real bug this fixes: an exact-title/
    # ordinal answer to a "which one?" cart-add clarification used to have
    # no durable state behind it at all, so it fell straight to understand()
    # with no memory of the question -- and a bare product title with no
    # cart-verb/cart-word could be (mis)classified as a brand-new search.
    if pending is not None and pending.kind == "cart_add_pending":
        result = resolve_pending_cart_add(
            session_id, message, activity, pending, state_manager, activity_store, trace,
        )
        if result is not None:
            return result

    # A3: social fast path (Sprint 6) -- greeting/how-are-you/gratitude
    # answered deterministically, entirely BEFORE understand(): zero LLM,
    # validation, search, recommendation, or embedding calls (see
    # src/scenario/social.py's classify_social(), a tiny bounded exact-
    # match set, never a substring/keyword check -- "hello, I need milk"
    # does not match and falls through to the normal workflow below).
    # Read-only, like OFF_TOPIC: counts the turn but never touches
    # history/turns/StateManager, so it can never corrupt a later memory
    # recall or mutate the active search.
    social_kind = classify_social(message)
    if social_kind is not None:
        activity_store.increment_turn_count(session_id)
        trace.route = "social_fast_path"
        state = state_manager.get_state(session_id)
        if social_kind == GREETING:
            reply = messages.social_greeting_response(message)
        elif social_kind == HOW_ARE_YOU:
            reply = messages.social_how_are_you_response(message)
        elif social_kind == NAME_QUESTION:
            reply = messages.social_name_response(message)
        else:  # GRATITUDE -- early vs late wording from EXISTING session
            # activity only (activity.turns: real shopping/memory turns so
            # far), never an LLM call or new sentiment signal.
            reply = messages.social_gratitude_response(message, had_prior_activity=bool(activity.turns))
        return _response(
            session_id, None, "social", state,
            user_message=message, assistant_message_override=reply,
        )

    # A4: deterministic zero-LLM cart fast path (LLM-usage audit task) --
    # an unmistakable bounded EN/AR cart view/clear command, or an add
    # reference naming an EXACT full shown/recent product title, needs no
    # language understanding at all; see src/scenario/cart_flow.py's
    # resolve_deterministic_cart_fast_path() for the exact, narrow
    # conditions under which it resolves (never guesses) -- anything less
    # unmistakable returns None here and falls through to understand()
    # below exactly as before.
    fast_path_state = state_manager.get_state(session_id)
    fast_path_result = resolve_deterministic_cart_fast_path(
        session_id, message, activity, activity_store, trace, fast_path_state,
    )
    if fast_path_result is not None:
        return fast_path_result

    # B: Understanding -- the ONLY history owner passed in is activity.history.
    # Sprint 6 product-selection follow-up: tells the REAL understand() (via
    # a contextvar, not a new understand_fn parameter -- every existing
    # stubbed understand_fn in tests keeps working unchanged) whether
    # anything is currently selectable, so it can tell "the Juhayna one"
    # (a selection) apart from "show me Juhayna products" (a new search).
    # Safe to call even when understand_fn is a test stub that ignores it.
    set_selectable_products_shown(bool(activity.last_shown_products))
    # Deterministic conversation-state/compact-memory task: compact, read-
    # only session context (src/scenario/llm_context.py) -- set fresh on
    # EVERY turn that reaches this ONE real call site (None when there's
    # nothing useful yet), so a stale value can never leak from a prior
    # turn. `pending` here is whatever A2.5/A2.5d already popped above but
    # could NOT resolve deterministically -- the LLM needs to know that
    # exact question (and its candidates) is still open. Same per-call
    # contextvar convention as set_selectable_products_shown() just above,
    # so every existing stubbed understand_fn in tests is unaffected.
    session_context = build_llm_session_context(activity, fast_path_state, pending, activity_store, session_id)
    set_session_context(session_context)
    trace.session_context_sent = session_context is not None
    trace.llm_calls = 1  # an attempted call counts even if it raises below --
    # real token usage (if any) is popped by the outer handle_message() wrapper.
    try:
        action = understand_fn(message, history=activity.history)
    except Exception as exc:
        # Provider errors (rate limits, quota, auth, schema/parse failures)
        # must stay observable server-side even though the user only ever
        # sees the generic fallback -- log type/status/message only, never
        # the request body or any credential.
        status = getattr(exc, "status_code", None)
        logger.warning("understand() failed: %s (status=%s): %s", type(exc).__name__, status, exc)
        trace.route = "error"
        # State-hardening task (issue 3): `pending` was already popped in
        # A2.5 above -- every resolver there returned None (no deterministic
        # evidence), meaning it's genuinely still an OPEN question this turn
        # was trying to answer via the LLM. An LLM failure must not silently
        # destroy it -- restore the EXACT original pending action (kind +
        # candidate ids) so the user's next real attempt can still answer
        # it, instead of falling through to a fresh, context-less search.
        if pending is not None:
            activity_store.set_pending_action(session_id, pending.kind, pending.shown_product_ids)
        state = state_manager.get_state(session_id)
        return _response(session_id, messages.understanding_failed(), "error", state, user_message=message)

    # C: deterministic Action normalization -- the approved Part 2 helper
    # plus the single recommendation-intent normalization function.
    action = normalize_target_facet(action)
    current_state = state_manager.get_state(session_id)
    action = normalize_recommendation_intent(action, current_state)
    _trace_set_action(trace, action)

    # C2: memory recall (Sprint 6) -- deterministic, read-only, zero extra LLM
    # calls, checked before OFF_TOPIC/NO_OP so a genuine "what did I say
    # before?" question never gets the generic off-topic refocus just because
    # it names no product. Reads `activity.turns`/StateManager BEFORE this
    # turn is appended/counted below -- the current recall question itself
    # must never become the recalled message.
    if action.memory_recall is not None:
        prior_turns = activity.turns
        answer = messages.memory_recall_answer(
            message, action.memory_recall,
            previous_message=prior_turns[-1].message if prior_turns else None,
            first_message=prior_turns[0].message if prior_turns else None,
            category=current_state.category, brand=current_state.brand,
            price_min=current_state.price_min, price_max=current_state.price_max,
        )
        activity_store.append_history(session_id, message, action.intent.value)
        activity_store.increment_turn_count(session_id)
        trace.route = "memory_recall"
        return _response(
            session_id, None, action.intent.value, current_state,
            user_message=message, assistant_message_override=answer,
        )

    # C3: cart commands + product selection, C4: checkout (cart/direct-
    # checkout architecture task) -- handled together so "I'll take the
    # second one and checkout" resolves the NEW selection FIRST, then
    # starts checkout from it in the same turn (all strictly deterministic:
    # the LLM only ever signals INTENT -- which reference, whether
    # checkout/cart action was asked for -- never a product id, cart
    # membership, price, or a completed order). No search, no
    # RecommendationService, no second LLM call, for any branch below.
    #
    # THREE domain concepts stay separate (never conflated): SHOWN PRODUCT
    # CONTEXT (last_shown_products/recent_shown_pages -- temporary
    # conversational selection authority), CART (activity.cart -- durable
    # within the session), CHECKOUT SELECTION (activity.selected_products --
    # the exact product(s)+quantity being purchased NOW, set only at the
    # instant a checkout draft actually opens). A plain, non-checkout
    # selection below never writes selected_products/cart -- it only
    # acknowledges a shown-product reference.
    #
    # MULTI-PRODUCT CHECKOUT REQUIRES CART: a checkout request naming 2+
    # products that are NOT already all in the cart may only ever become a
    # pending add-to-cart proposal (see A2.5b above), never a multi-line
    # DIRECT draft -- the cart is the only durable authority for buying
    # multiple products together. When the named products ARE already all
    # in the cart (correction), checkout opens directly as a cart-subset
    # draft instead of pointlessly re-asking to add them -- see the
    # cart-reconciliation logic in the has_selection_reference branch below.
    has_selection_reference = (
        bool(action.selection_ordinals) or action.selection_last or action.selection_text
    )
    # A checkout request left unresolved by an ambiguous selection survives
    # as a PendingAction (`pending`, popped in A2.5 above) into the next
    # NO_OP turn -- gated on NO_OP so a genuine new search/filter turn is
    # never hijacked into checkout.
    effective_checkout_request = action.checkout_request or (
        action.intent == Intent.NO_OP and pending is not None and pending.kind == "checkout_pending_selection"
    )
    # A bare cart-checkout clarification (CASE 3) is resolved entirely
    # deterministically, BEFORE understand() is ever called -- see A2.5c
    # above. Reaching this point with `pending` still at that kind means
    # A2.5c already tried and could not safely resolve it (zero raw
    # evidence at all), so it is intentionally NOT re-attempted here with
    # the LLM's Action -- that would be a second, redundant resolver and
    # would break the "0 Groq calls" guarantee A2.5c exists to provide.
    cart_action = action.cart_action
    # Deterministic conversation-state task: the SAME "pending survives a
    # NO_OP continuation" pattern as effective_checkout_request just above,
    # for a pending cart-ADD instead of checkout -- A2.5d already tried and
    # could not resolve it from raw evidence alone (an exact title/ordinal/
    # "last" would have already returned there with 0 LLM calls); reaching
    # here means the LLM's semantic read (helped by the compact pending
    # context built above) is what identifies the answer, but the ADD
    # itself is still forced deterministically by the backend, never
    # inferred from cart_action alone (the model isn't asked to restate it
    # -- see the SESSION MEMORY prompt guidance in understanding/llm.py).
    # Gated on NO_OP so a genuine new search/topic switch (the LLM's own
    # job to recognize, given that same context) is never hijacked into a
    # cart mutation.
    if (
        cart_action is None and pending is not None and pending.kind == "cart_add_pending"
        and action.intent == Intent.NO_OP and (has_selection_reference or action.generic_reference is not None)
    ):
        cart_action = "add"

    if cart_action is not None or has_selection_reference or effective_checkout_request:
        activity_store.append_history(session_id, message, action.intent.value)
        activity_store.increment_turn_count(session_id)
        state = state_manager.get_state(session_id)

        # --- C3: cart commands -- resolved deterministically; the LLM only
        # signals WHICH operation (cart_action) and WHICH item(s)
        # (selection_ordinals/last/text, reused unchanged). See
        # src/scenario/cart_flow.py's handle_cart_action() -- exhaustive
        # over cart_action's 5 literals, always returns a real response. ---
        if cart_action is not None:
            return handle_cart_action(
                cart_action, session_id, message, state, activity, action, activity_store, trace, pending=pending,
            )

        # --- plain product selection: acknowledges a shown-product
        # reference without opening a checkout draft or touching cart/
        # selected_products (SHOWN PRODUCT CONTEXT only). ---
        if has_selection_reference and not effective_checkout_request:
            trace.route = "product_selection"
            recent_shown = activity_store.recent_shown_products(session_id)
            products, reason = _trace_resolve(
                trace, "last_shown_products+recent_shown", activity.last_shown_products, recent_shown, action, message,
            )
            if reason != "ok":
                reply = _selection_failure_message(message, reason, len(activity.last_shown_products))
            else:
                reply = messages.product_selected_response(message, [(p.name_en, p.name_ar) for p in products])
            return _response(
                session_id, None, "product_selection", state,
                user_message=message, assistant_message_override=reply,
            )

        # --- C4: checkout. effective_checkout_request from here on (a bare
        # cart-checkout scope clarification is fully handled by A2.5c,
        # deterministically, before understand() is ever called -- see
        # above; it never reaches this point). trace.route is set
        # individually at each actual outcome below (correction: no more
        # premature blanket "checkout_start" default here that could leak
        # into a clarification response when resolution fails -- see
        # _checkout_from_lines(), which now sets it only on its own real
        # success). ---
        if has_selection_reference and _raw_message_mentions_cart(message):
            # "checkout Pepsi and Sardine from my cart" -- explicit cart
            # subset/all, resolved against the cart, never shown products.
            cart_pool = [line.product for line in activity.cart]
            products, reason = _trace_resolve(trace, "cart", cart_pool, cart_pool, action, message)
            if reason != "ok":
                if effective_checkout_request:
                    # State-hardening task (issues 1/2): this question is
                    # explicitly about the CART (_raw_message_mentions_cart)
                    # -- the frozen candidate snapshot must be the cart pool
                    # actually just searched, never last_shown_products.
                    activity_store.set_pending_action(
                        session_id, "checkout_pending_selection", shown_product_ids=[p.id for p in cart_pool],
                    )
                trace.route = "checkout_clarification"
                return _response(
                    session_id, None, "checkout_clarification", state, user_message=message,
                    assistant_message_override=_selection_failure_message(
                        message, reason, len(cart_pool), cart=True,
                    ),
                )
            by_id = {line.product.id: line for line in activity.cart}
            selected_lines = [by_id[p.id] for p in products]
            return _checkout_from_lines(
                session_id, message, state, selected_lines, activity_store, trace,
                cart_ids=[line.product.id for line in selected_lines],
            )

        if has_selection_reference:
            # DIRECT checkout from shown/cart authority. Named matching
            # (Correction: MULTI-PRODUCT PURCHASE = CART ONLY, reconciled
            # with what's ALREADY in the cart) uses the combined cart+
            # recent-shown pool -- positional/vague references are
            # UNAFFECTED (they only ever consult current_shown, never this
            # pool -- see _cart_and_recent_pool()'s own docstring).
            named_pool = _cart_and_recent_pool(activity, activity_store, session_id)
            products, reason = _trace_resolve(
                trace, "last_shown_products+cart+recent_shown", activity.last_shown_products, named_pool, action, message,
            )
            if reason != "ok":
                # Correction: a SINGLE textual reference (no ordinal/last
                # signal at all -- has_selection_reference guarantees
                # selection_text is what's set here) that failed above may
                # still refer to a real cart line that wasn't recently
                # shown (e.g. "checkout Pepsi" while browsing something
                # unrelated). Retry the SAME resolver, scoped to the cart
                # pool -- its own selection_text exact-then-fuzzy fallback
                # (normally current_shown-only) then gets a fair shot
                # against real cart products. NEVER for a positional/vague
                # reference outside explicit cart context -- ordinals/last
                # must still mean current_shown only, so this is gated
                # strictly on there being no ordinal/last signal.
                cart_wording = False
                if not action.selection_ordinals and not action.selection_last and activity.cart:
                    cart_pool = [line.product for line in activity.cart]
                    cart_products, cart_reason = _trace_resolve(trace, "cart", cart_pool, cart_pool, action, message)
                    if cart_reason == "ok":
                        products, reason = cart_products, "ok"
                    elif cart_reason == "ambiguous":
                        reason, cart_wording = "ambiguous", True
                if reason != "ok":
                    # State-hardening task (issues 1/2): freeze whichever
                    # pool this attempt actually searched -- the cart pool
                    # when the retry above ran against it (cart_wording),
                    # else the shown+cart+recent pool from the resolve
                    # above (named_pool is already a superset of
                    # last_shown_products -- see _cart_and_recent_pool()).
                    candidate_pool = cart_pool if cart_wording else named_pool
                    activity_store.set_pending_action(
                        session_id, "checkout_pending_selection",
                        shown_product_ids=[p.id for p in candidate_pool],
                    )
                    trace.route = "checkout_clarification"
                    return _response(
                        session_id, None, "checkout_clarification", state, user_message=message,
                        assistant_message_override=_selection_failure_message(
                            message, reason,
                            len(activity.cart) if cart_wording else len(activity.last_shown_products),
                            cart=cart_wording,
                        ),
                    )

            # Reconcile against the cart: a resolved product already in the
            # cart is checked out AS that cart line (its real quantity),
            # never re-added/incremented; a resolved product NOT in the
            # cart still needs one (see below) before it can join a
            # multi-product purchase.
            cart_by_id = {line.product.id: line for line in activity.cart}
            missing = [p for p in products if p.id not in cart_by_id]

            if not missing:
                # Every requested product is already a real cart line --
                # open a cart-subset checkout directly, preserving each
                # line's own quantity. Correct for 1 product too (a single
                # "checkout X" that happens to already be in the cart
                # checks out that cart line, not a separate direct draft).
                selected_lines = [cart_by_id[p.id] for p in products]
                return _checkout_from_lines(
                    session_id, message, state, selected_lines, activity_store, trace,
                    cart_ids=[p.id for p in products],
                )

            if len(products) == 1:
                # The one requested product isn't in the cart -- ordinary
                # DIRECT checkout, cart untouched on success.
                return _checkout_from_lines(
                    session_id, message, state, [CartLine(product=products[0], quantity=1)],
                    activity_store, trace, cart_ids=[],
                )

            # 2+ products requested, at least one missing from the cart --
            # the cart is the only durable authority for buying multiple
            # products together, so propose adding ONLY the missing
            # ones (never re-proposing/re-adding what's already there).
            trace.route = "multi_direct_requires_cart"
            activity_store.set_pending_action(
                session_id, "multi_add_to_cart_pending", shown_product_ids=[p.id for p in missing],
            )
            return _response(
                session_id, None, "checkout_clarification", state, user_message=message,
                assistant_message_override=messages.multi_direct_needs_cart_response(
                    message, [(p.name_en, p.name_ar) for p in missing],
                ),
            )

        # Bare "checkout"/"حاسب" -- no product named, no cart mention:
        # deterministically driven by CART STATE (never a stale prior
        # selection, never an empty search).
        cart_lines = activity.cart
        if not cart_lines:
            trace.route = "checkout_clarification"
            return _response(
                session_id, None, "checkout_clarification", state, user_message=message,
                assistant_message_override=messages.cart_checkout_empty_response(message),
            )
        if len(cart_lines) == 1:
            return _checkout_from_lines(
                session_id, message, state, cart_lines, activity_store, trace,
                cart_ids=[cart_lines[0].product.id],
            )
        if _raw_message_supports_checkout_all(message):
            return _checkout_from_lines(
                session_id, message, state, cart_lines, activity_store, trace,
                cart_ids=[line.product.id for line in cart_lines],
            )
        activity_store.set_pending_action(
            session_id, "cart_checkout_pending", shown_product_ids=[line.product.id for line in cart_lines],
        )
        trace.route = "checkout_clarification"
        return _response(
            session_id, None, "checkout_clarification", state, user_message=message,
            assistant_message_override=messages.cart_checkout_clarify_response(
                message, [(line.product.name_en, line.product.name_ar, line.quantity) for line in cart_lines],
            ),
        )

    # D: OFF_TOPIC
    if action.intent == Intent.OFF_TOPIC:
        activity_store.increment_turn_count(session_id)
        trace.route = "off_topic"
        state = state_manager.get_state(session_id)
        return _response(
            session_id, messages.off_topic_refocus(action.customer_service_tone),
            action.intent.value, state, user_message=message,
        )

    # E: NO_OP / pure recommendation-exploration -- StateManager.apply() is
    # never called for this intent (state_manager.py raises ValueError for
    # it -- it is not one of its five handled intents by design).
    if action.intent == Intent.NO_OP:
        outcome = derive_recommendation_search(current_state, activity)
        activity_store.append_history(session_id, message, action.intent.value)
        activity_store.increment_turn_count(session_id)
        state = state_manager.get_state(session_id)  # unchanged canonical state
        trace.route = "recommendation"

        if outcome.needs_clarification:
            return _response(
                session_id, messages.recommendation_needs_clarification(), action.intent.value, state,
                user_message=message, recommendation_request=action.recommendation_request,
                needs_clarification=True,
            )

        # Sprint 6 recommendation task: ALWAYS derived from the CANONICAL
        # active state (current_state), never the temporary relaxed price
        # band below -- a recommendation must never quietly reflect a
        # loosened constraint just because the exploration policy widened
        # one for `products`. The two mechanisms are independent by design
        # (see src/recommendation/service.py's module docstring).
        #
        # A pure continuation (no relaxation applied, not exhausted) hasn't
        # changed current_state at all since it was last actually searched
        # -- that search's own results are still sitting in
        # last_shown_products, so reusing them here avoids re-running the
        # identical SearchAdapter call just to rediscover the same ids
        # (real symptom: "suggest to me more" after a plain "pepsi"/"pebsi"
        # search re-searched "pepsi" a third time for no new information).
        # Exhaustion and a genuine new relaxation band both still need a
        # fresh search: by then last_shown_products may hold a DIFFERENT
        # (relaxed-band or previously-recommended) list, not current_state's
        # own products.
        reused_shown_ids = (
            [p.id for p in activity.last_shown_products]
            if not outcome.relaxation_applied and outcome.effective_filters is not None
            else None
        )
        recommendations = _build_recommendations(
            action, current_state, _counted_search, limit, trace, known_shown_ids=reused_shown_ids,
        )
        # Sprint 6 product-selection task: recommendations shown here become
        # the latest selectable context, same rule as the F branch below.
        if recommendations:
            activity_store.set_last_shown_products(session_id, _to_displayed_products(recommendations))

        if outcome.effective_filters is None:
            # relaxation_exhausted -- no further automatic band, no SearchAdapter call
            return _response(
                session_id, messages.recommendation_relaxation_exhausted(), action.intent.value, state,
                user_message=message, recommendations=recommendations,
                recommendation_request=action.recommendation_request,
            )

        if not outcome.relaxation_applied:
            # Sprint 6 recommendation-context task: NOTHING new was explored
            # this turn (effective_filters is just the unchanged canonical
            # state) -- a pure "recommend me more"/"show me other options"
            # continuation, not a fresh search. Re-running and re-displaying
            # the SAME products the user already has is noise ahead of the
            # actual ask (real symptom: "Sure, I found these products for
            # you, and I also found some options you may like," followed by
            # a redundant product list before "Recommended for you"). No
            # SearchAdapter call needed here at all -- recommendations alone
            # answer this turn. A genuinely NEW price band (relaxation_
            # applied=True, below) is the one case with real new content to
            # show as `products`, distinct from the recommendations.
            return _response(
                session_id, None, action.intent.value, state,
                user_message=message, recommendations=recommendations,
                recommendation_request=action.recommendation_request,
            )

        result = _counted_search(outcome.effective_filters, limit, offset)
        parts, likely_ooc = _search_message_parts(
            action, outcome.effective_filters, result, activity,
            relaxation_note=outcome.relaxation_applied,
        )
        if not recommendations and result.products:
            activity_store.set_last_shown_products(session_id, _to_displayed_products(result.products))
        return _response(
            session_id, " ".join(parts) or None, action.intent.value, state,
            user_message=message,
            products=result.products, total_count=result.total_count,
            zero_result=result.zero_result, likely_out_of_catalog=likely_ooc,
            effective_price_min=outcome.effective_filters.price_min,
            effective_price_max=outcome.effective_filters.price_max,
            recommendations=recommendations,
            recommendation_request=action.recommendation_request,
        )

    # G: RESET special case -- clears the CANONICAL shopping state (via the
    # normal apply() path, same as any other intent) and this session's
    # exploration bookkeeping, but Sprint 6 lifecycle task: deliberately does
    # NOT wipe conversation memory (history/turns) -- a "start over" on the
    # active search is not the same event as the session ending. It IS a
    # real accepted turn like any other, so it appends/counts normally
    # (full session teardown, including memory, is handled separately -- see
    # A1's expiration check and SessionActivityStore.reset()).
    if action.intent == Intent.RESET:
        trace.route = "reset"
        resolved = validate(action)
        trace.validation_calls += 1

        # Correction: this cleanup happens ONCE here, before either branch
        # below, not duplicated in (or missing from) one of them -- a
        # genuine "start over" must clear checkout/recent-shown/cart
        # REGARDLESS of whether the newly requested concept turns out to
        # be catalog-supported (real bug: the out-of-catalog branch used to
        # return before ever reaching this cleanup, so a cart could survive
        # "start over and find <unsupported concept>"). `had_checkout`
        # captured BEFORE clearing so the response flag reflects a REAL
        # invalidation. Unlike a normal new search/topic switch (F branch),
        # which deliberately keeps recent-shown memory and NEVER touches
        # the cart -- RESET forgets both, a strictly bigger wipe.
        had_checkout = activity.checkout_active
        activity_store.clear_checkout(session_id)
        activity_store.clear_recent_shown_products(session_id)
        activity_store.clear_cart(session_id)

        # Sprint 6 out-of-catalog grounding task: same pre-search gate as
        # the F branch below -- a "start over, and get me <unsupported
        # concept>" must still start over, just never populate the fresh
        # state with the unsupported concept (price bounds, if any, are
        # unrelated to catalog membership and are kept).
        if _lacks_catalog_evidence(resolved):
            trace.route = "out_of_catalog"
            state_manager.apply(
                session_id, action,
                FilterSet(price_min=resolved.price_min, price_max=resolved.price_max),
            )
            activity_store.append_history(session_id, message, action.intent.value)
            activity_store.increment_turn_count(session_id)
            activity_store.reset_relaxation_step(session_id)
            new_state = state_manager.get_state(session_id)
            parts = _out_of_catalog_message_parts(action, resolved, activity_store.get(session_id))
            return _response(
                session_id, " ".join(parts) or None, action.intent.value, new_state,
                user_message=message, zero_result=True, likely_out_of_catalog=True,
                cart=_to_cart_out([]), checkout_invalidated=had_checkout,
            )

        state_manager.apply(session_id, action, resolved)
        activity_store.append_history(session_id, message, action.intent.value)
        activity_store.increment_turn_count(session_id)
        activity_store.reset_relaxation_step(session_id)  # the search itself was
        # just cleared -- any prior automatic price-relaxation band no longer applies
        new_state = state_manager.get_state(session_id)
        result = _counted_search(new_state, limit, offset)
        parts, likely_ooc = _search_message_parts(action, new_state, result, activity_store.get(session_id))
        if result.products:
            activity_store.set_last_shown_products(session_id, _to_displayed_products(result.products))
        return _response(
            session_id, " ".join(parts) or None, action.intent.value, new_state,
            user_message=message,
            products=result.products, total_count=result.total_count,
            zero_result=result.zero_result, likely_out_of_catalog=likely_ooc,
            cart=_to_cart_out([]), checkout_invalidated=had_checkout,
        )

    # F: normal state-changing intents (SEARCH/ADD_FILTER/MODIFY_FILTER/
    # REMOVE_FILTER) -- the EXISTING deterministic validate()/StateManager
    # merge semantics, never duplicated here. An explicit exception, not a
    # bare `assert` -- `python -O` strips asserts silently (confirmed
    # project-wide in the Sprint 0-3 ingestion fail-safe pass), and this
    # exhaustiveness check over schema.py's Intent enum must actually fire.
    if action.intent not in _STATE_CHANGING_INTENTS:
        raise ValueError(f"unhandled effective intent reached orchestration: {action.intent!r}")
    resolved = validate(action)
    trace.validation_calls += 1

    # Sprint 6 out-of-catalog grounding task: deterministic pre-search gate
    # -- checked against THIS TURN's own resolution, before StateManager
    # ever merges it, so a prior turn's already-active category can never
    # paper over a brand-new unsupported concept, and an unsupported
    # request never overwrites/merges into the active search at all.
    # Skips SearchAdapter, vector fallback, and RecommendationService
    # entirely -- see _lacks_catalog_evidence().
    if _lacks_catalog_evidence(resolved):
        trace.route = "out_of_catalog"
        activity_store.append_history(session_id, message, action.intent.value)
        activity_store.increment_turn_count(session_id)
        # Correction: a genuine new search/topic request must invalidate a
        # previously-open checkout draft even when the requested concept
        # turns out to be out-of-catalog (real bad state: open a Pepsi
        # checkout, then ask for an out-of-catalog "laptop charger" -- the
        # old Pepsi draft used to survive, since this early return sat
        # BEFORE the normal-search clear_checkout() logic further below).
        # `had_checkout` captured BEFORE clearing so the response flag
        # reflects a REAL invalidation. Cart/recent-shown/StateManager's
        # existing search state are deliberately left untouched -- an OOC
        # search still must not mutate canonical filters, exactly as today.
        had_checkout = activity.checkout_active
        activity_store.clear_checkout(session_id)
        state = state_manager.get_state(session_id)  # unchanged canonical state
        parts = _out_of_catalog_message_parts(action, resolved, activity)
        return _response(
            session_id, " ".join(parts) or None, action.intent.value, state,
            user_message=message, zero_result=True, likely_out_of_catalog=True,
            checkout_invalidated=had_checkout,
        )

    trace.route = "shopping_search"
    state_manager.apply(session_id, action, resolved)
    activity_store.append_history(session_id, message, action.intent.value)
    activity_store.increment_turn_count(session_id)
    activity_store.reset_relaxation_step(session_id)  # real state change -> any prior
    # automatic exploration band no longer applies
    had_checkout = activity.checkout_active  # captured BEFORE clearing below, so the
    # response flag (checkout_invalidated) reflects a REAL invalidation
    activity_store.clear_checkout(session_id)  # Sprint 6 checkout task: a new normal
    # search/topic switch ends any open checkout draft -- never stale into a new flow
    # (the cart itself is deliberately left untouched -- see clear_checkout()'s own
    # docstring; a normal search must NEVER clear the cart)

    new_state = state_manager.get_state(session_id)
    result = _counted_search(new_state, limit, offset)
    parts, likely_ooc = _search_message_parts(action, new_state, result, activity)

    # Sprint 6 recommendation task: reuses the search that JUST ran under
    # new_state -- no second SearchAdapter call needed here, unlike the
    # NO_OP path above. [] whenever recommendation_request is false.
    recommendations: list[dict] = []
    if action.recommendation_request and result.products:
        shown_ids = [p["id"] for p in result.products]
        trace.recommendation_calls += 1
        recommendations = get_recommendation_service().recommend(
            new_state.query_text, shown_ids, new_state, limit=limit,
        )

    # Sprint 6 product-selection task: whatever was actually shown LAST this
    # turn becomes the selectable list -- recommendations (shown after/
    # alongside products) take priority when both are present, matching
    # visual/conversational recency. A turn that shows nothing new never
    # reaches here with an empty list (this whole F branch already returned
    # early above for zero-result/out-of-catalog), so the old list is never
    # wrongly cleared.
    if recommendations:
        activity_store.set_last_shown_products(session_id, _to_displayed_products(recommendations))
    elif result.products:
        activity_store.set_last_shown_products(session_id, _to_displayed_products(result.products))

    # Sprint 6 recommendation-offer task: after a successful NORMAL search
    # that did NOT itself already ask for recommendations, offer them --
    # conversationally only. RecommendationService is deliberately NOT
    # called here (trace.recommendation_calls stays 0 for this turn); only
    # the real shown product ids are remembered (PendingAction) for a
    # possible bare "yes" on the very next turn (see A2.5 above).
    assistant_message_override = None
    if not action.recommendation_request and not result.zero_result and not likely_ooc and result.products:
        activity_store.set_pending_action(session_id, "recommendation_offer", [p["id"] for p in result.products])
        base_reply = messages.build_assistant_message(
            user_message=message, intent=action.intent.value, recommendation_request=False,
            recommendations_count=0, zero_result=False, likely_out_of_catalog=False, blocked=False,
        )
        assistant_message_override = f"{base_reply} {messages.recommendation_offer_prompt(message)}"

    return _response(
        session_id, " ".join(parts) or None, action.intent.value, new_state,
        user_message=message,
        products=result.products, total_count=result.total_count,
        zero_result=result.zero_result, likely_out_of_catalog=likely_ooc,
        recommendations=recommendations,
        recommendation_request=action.recommendation_request,
        assistant_message_override=assistant_message_override,
        checkout_invalidated=had_checkout,
    )


_state_manager: StateManager | None = None
_activity_store: SessionActivityStore | None = None


def get_state_manager() -> StateManager:
    global _state_manager
    if _state_manager is None:
        _state_manager = StateManager()
    return _state_manager


def get_activity_store() -> SessionActivityStore:
    global _activity_store
    if _activity_store is None:
        _activity_store = SessionActivityStore()
    return _activity_store
