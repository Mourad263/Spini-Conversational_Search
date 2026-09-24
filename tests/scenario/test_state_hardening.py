"""Focused state-hardening follow-up to the deterministic conversation-
state / compact-LLM-memory architecture task. Fixes exactly 4 issues, all
reusing `PendingAction.shown_product_ids` -- no new state, no new
mechanism, no architecture change.

1. A pending selection stores its exact candidate snapshot at creation
   time (`PendingAction.shown_product_ids`) and the next turn resolves
   only against those stored ids -- never a re-derived
   last_shown_products/cart read.
2. The deterministic resolver and the compact LLM CANDIDATES block use
   the SAME helper (`llm_context.resolve_pending_candidates()`) to turn
   those ids back into real products -- they can never disagree about
   what's being disambiguated. A cart-scoped question's candidates come
   from the cart, never last_shown_products.
3. An LLM failure (429/network/schema/provider error) during an LLM-
   assisted pending continuation restores the exact original pending
   action instead of silently dropping it.
4. `understand()` never sends the same recent turns twice (compact
   session context's RECENT_CONTEXT vs. the raw `history` list).

Hermetic -- `understand_fn` is always stubbed (or, for the history-
duplication test, the real `understand()` is exercised against a fake
OpenAI client -- still zero real network/API calls).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.scenario.llm_context import build_llm_session_context, resolve_pending_candidates  # noqa: E402
from src.scenario.orchestrator import handle_message  # noqa: E402
from src.scenario.session_activity import DisplayedProduct, SessionActivityStore  # noqa: E402
from src.search_adapter.adapter import FilterSet, SearchAdapter, SearchResult  # noqa: E402
from src.state_manager.state_manager import StateManager  # noqa: E402
from src.understanding.schema import Action, Intent  # noqa: E402


def stub_understand(*actions: Action):
    it = iter(actions)

    def _stub(message: str, history: list[str] | None = None) -> Action:
        return next(it)

    return _stub


class RecordingSearch:
    def __init__(self) -> None:
        self.calls: list[FilterSet] = []

    def __call__(self, filters: FilterSet, limit: int, offset: int) -> SearchResult:
        self.calls.append(filters)
        return SearchAdapter().search(filters, limit=limit, offset=offset)


_A = DisplayedProduct(id=101, name_en="Beef Mince Low Fat Balady Premium",
                       name_ar="Lahma Low Fat Premium", price=189.0, special_price=189.0)
_B = DisplayedProduct(id=102, name_en="Minced Beef Low Fat Balady",
                       name_ar="Lahma Low Fat", price=175.0, special_price=175.0)
_C = DisplayedProduct(id=103, name_en="Chicken Breast Boneless",
                       name_ar="Sadr Farkh", price=120.0, special_price=120.0)


def _primed_session(session_id: str, shown: list[DisplayedProduct]) -> tuple[StateManager, SessionActivityStore]:
    sm, store = StateManager(), SessionActivityStore()
    store.set_last_shown_products(session_id, shown)
    return sm, store


# ======================================================================
# Issue 1 -- frozen candidate snapshot survives a later shown-products
# change.
# ======================================================================

def test_pending_candidate_snapshot_survives_last_shown_products_change():
    sm, store = _primed_session("s1", [_A, _B])
    store.set_pending_action("s1", "cart_add_pending", shown_product_ids=[_A.id, _B.id])

    # Something else changes the "current" shown context entirely (a
    # different product C) WITHOUT going through the normal pending-
    # clearing flow -- the pending question's own candidate set must be
    # unaffected by this.
    store.set_last_shown_products("s1", [_C])
    assert store.get("s1").pending_action.shown_product_ids == [_A.id, _B.id]

    search = RecordingSearch()

    def _hostile(message, history=None):
        raise AssertionError("understand() must not be called for an exact-title pending continuation")

    handle_message("s1", _A.name_en, state_manager=sm, activity_store=store,
                    understand_fn=_hostile, search_fn=search)

    cart = store.get("s1").cart
    assert len(cart) == 1
    assert cart[0].product.id == _A.id  # resolved against the FROZEN snapshot, not the new [_C] context
    assert store.get("s1").pending_action is None
    assert len(search.calls) == 0


# ======================================================================
# Issue 1/2 -- checkout pending uses CART candidates when the question is
# cart-scoped, never last_shown_products.
# ======================================================================

def test_checkout_pending_uses_cart_candidates_not_shown_products():
    sm, store = StateManager(), SessionActivityStore()
    # Two DIFFERENT beef products are in the cart, while a THIRD (chicken)
    # is merely displayed -- the checkout question below explicitly
    # mentions "my cart", so its candidates must be the cart's own two
    # products, never the displayed chicken.
    from src.scenario.session_activity import CartLine
    store.get("s1").cart = [CartLine(product=_A, quantity=1), CartLine(product=_B, quantity=1)]
    store.set_last_shown_products("s1", [_C])
    search = RecordingSearch()

    # A cart-mentioning checkout reference that fails to resolve uniquely
    # (selection_text "Low Fat" fuzzy-matches both cart lines).
    understand_fn = stub_understand(
        Action(intent=Intent.NO_OP, selection_text="Low Fat", checkout_request=True),
    )
    r = handle_message("s1", "checkout the low fat one from my cart", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert r.intent == "checkout_clarification"
    pending = store.get("s1").pending_action
    assert pending is not None
    assert pending.kind == "checkout_pending_selection"
    assert set(pending.shown_product_ids) == {_A.id, _B.id}  # cart ids, never _C (shown, not cart)


# ======================================================================
# Issue 2 -- the deterministic resolver and the compact LLM CANDIDATES
# block resolve the SAME candidate set from the same stored ids.
# ======================================================================

def test_resolver_and_llm_context_use_the_same_candidate_set():
    sm, store = _primed_session("s1", [_A, _B])
    store.set_pending_action("s1", "cart_add_pending", shown_product_ids=[_A.id, _B.id])
    activity = store.get("s1")
    pending = activity.pending_action

    resolver_candidates = resolve_pending_candidates(activity, store, "s1", pending)
    context = build_llm_session_context(activity, FilterSet(), pending, store, "s1")

    assert [p.id for p in resolver_candidates] == [_A.id, _B.id]
    assert context is not None
    for p in resolver_candidates:
        assert p.name_en in context


def test_llm_context_checkout_pending_candidates_come_from_cart_ids():
    """Directly proves llm_context no longer branches on pending.kind to
    pick last_shown_products vs cart -- it resolves whatever ids the
    pending action itself stored, exactly like the resolver does."""
    from src.scenario.session_activity import CartLine

    activity_store = SessionActivityStore()
    activity_store.get("s1").cart = [CartLine(product=_A, quantity=2)]
    activity_store.set_last_shown_products("s1", [_C])  # a DIFFERENT, irrelevant shown product
    activity_store.set_pending_action("s1", "checkout_pending_selection", shown_product_ids=[_A.id])
    activity = activity_store.get("s1")
    pending = activity.pending_action

    context = build_llm_session_context(activity, FilterSet(), pending, activity_store, "s1")

    assert context is not None
    assert _A.name_en in context
    assert _C.name_en not in context  # never the unrelated shown product


# ======================================================================
# Issue 3 -- an LLM failure during a pending continuation must not
# destroy the pending state.
# ======================================================================

def test_llm_failure_preserves_pending_action_and_candidates():
    sm, store = _primed_session("s1", [_A, _B])
    store.set_pending_action("s1", "cart_add_pending", shown_product_ids=[_A.id, _B.id])
    search = RecordingSearch()

    def _raising(message, history=None):
        raise RuntimeError("simulated 429/provider error")

    # No raw exact-title/ordinal/"last" evidence at all -- the
    # deterministic resolver returns None, so this reaches the real
    # (here: raising) understand() call.
    r = handle_message("s1", "اللي قليل الدسم", state_manager=sm, activity_store=store,
                        understand_fn=_raising, search_fn=search)

    assert r.intent == "error"
    pending = store.get("s1").pending_action
    assert pending is not None
    assert pending.kind == "cart_add_pending"
    assert set(pending.shown_product_ids) == {_A.id, _B.id}  # exact original candidates preserved
    assert store.get("s1").cart == []  # no accidental mutation


# ======================================================================
# A genuine successful topic switch still clears pending normally (LLM
# call succeeded -- its own semantic result decides the transition).
# ======================================================================

def test_successful_topic_switch_still_clears_pending():
    sm, store = _primed_session("s1", [_A, _B])
    store.set_pending_action("s1", "cart_add_pending", shown_product_ids=[_A.id, _B.id])
    search = RecordingSearch()

    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk", recommendation_request=True),
    )
    r = handle_message("s1", "لا خلاص وريني لبن", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert r.intent == "search"
    assert store.get("s1").pending_action is None
    assert store.get("s1").cart == []
    assert len(search.calls) == 1


# ======================================================================
# Issue 4 -- understand() never sends the same recent turns twice.
# ======================================================================

class _FakeParsed:
    def __init__(self, action: Action) -> None:
        self.parsed = action


class _FakeChoice:
    def __init__(self, action: Action) -> None:
        self.message = _FakeParsed(action)


class _FakeCompletion:
    def __init__(self, action: Action) -> None:
        self.choices = [_FakeChoice(action)]
        self.usage = None


class _FakeCompletions:
    def __init__(self, captured: dict) -> None:
        self._captured = captured

    def parse(self, *, model, messages, response_format, temperature):
        self._captured["messages"] = messages
        return _FakeCompletion(Action(intent=Intent.NO_OP))


class _FakeChat:
    def __init__(self, captured: dict) -> None:
        self.completions = _FakeCompletions(captured)


class _FakeClient:
    def __init__(self, captured: dict) -> None:
        self.chat = _FakeChat(captured)


def test_understand_does_not_duplicate_recent_history_when_session_context_present():
    import src.understanding.llm as llm_module

    captured: dict = {}
    original_get_client = llm_module._get_client
    llm_module._get_client = lambda: _FakeClient(captured)
    try:
        llm_module.set_session_context(
            "CURRENT_CONTEXT\nphase: BROWSING\n\nRECENT_CONTEXT\nuser: turn A\noutcome: displayed products\n"
            "user: turn B\noutcome: cart add ambiguous",
        )
        llm_module.understand("current message", history=["turn A", "turn B"])
    finally:
        llm_module._get_client = original_get_client
        llm_module.set_session_context(None)

    sent = captured["messages"]
    user_contents = [m["content"] for m in sent if m["role"] == "user"]
    # The current message is always sent as its own user turn ...
    assert "current message" in user_contents
    # ... but the raw history turns must NOT be sent again as separate
    # user messages -- they're already inside the SESSION MEMORY system
    # message above.
    assert "turn A" not in user_contents
    assert "turn B" not in user_contents
    system_contents = " ".join(m["content"] for m in sent if m["role"] == "system")
    assert "turn A" in system_contents  # still present -- just once, in SESSION MEMORY


def test_understand_still_sends_raw_history_without_session_context():
    """Behavior for the common no-compact-context-yet case is unchanged."""
    import src.understanding.llm as llm_module

    captured: dict = {}
    original_get_client = llm_module._get_client
    llm_module._get_client = lambda: _FakeClient(captured)
    try:
        llm_module.set_session_context(None)
        llm_module.understand("current message", history=["turn A", "turn B"])
    finally:
        llm_module._get_client = original_get_client

    sent = captured["messages"]
    user_contents = [m["content"] for m in sent if m["role"] == "user"]
    assert "turn A" in user_contents
    assert "turn B" in user_contents
    assert "current message" in user_contents


# ======================================================================
# Regression: the original bug's exact-title pending cart-add continuation
# still resolves deterministically, 0 LLM calls, after this hardening pass.
# ======================================================================

def test_original_exact_title_pending_cart_add_still_resolves_with_zero_llm_calls():
    sm, store = _primed_session("s1", [_A, _B])
    store.set_pending_action("s1", "cart_add_pending", shown_product_ids=[_A.id, _B.id])
    search = RecordingSearch()

    def _hostile(message, history=None):
        raise AssertionError("understand() must not be called for an exact-title pending continuation")

    r = handle_message("s1", "Beef Mince Low Fat Balady Premium", state_manager=sm, activity_store=store,
                        understand_fn=_hostile, search_fn=search)

    cart = store.get("s1").cart
    assert len(cart) == 1
    assert cart[0].product.id == _A.id
    assert store.get("s1").pending_action is None
    assert len(search.calls) == 0
    assert "beef mince low fat balady premium" in r.assistant_message.lower()
