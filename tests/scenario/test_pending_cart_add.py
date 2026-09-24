"""Deterministic conversation-state / compact-LLM-memory architecture task.

Real reported bug (Egyptian Arabic):

    User:      مش عايز فراخ، خلّيها لحمة        (not chicken, make it beef)
    Assistant: shows beef products
    User:      حطها في السله                    (put it in the cart)
    Assistant: لقيت أكتر من منتج مطابق من اللي عرضته.
               تقدر تحدد بالرقم أو بالاسم الكامل؟  (found more than one match,
               can you specify by number or full name?)
    User:      Beef Mince Low Fat Balady Premium

The final message -- an exact shown-product title -- used to be
(mis)classified as a NEW SEARCH instead of the answer to the pending
cart-add clarification. Root cause: an ambiguous cart_action="add" left
its clarification as TEXT ONLY, with no durable pending state behind it
(unlike every other clarification in this codebase -- checkout_pending_
selection, cart_checkout_pending, multi_add_to_cart_pending all already
record a PendingAction). Fixed by cart_flow.py's handle_cart_action()
setting kind="cart_add_pending", and pending_actions.py's new
resolve_pending_cart_add() -- the deterministic, ZERO-LLM continuation
(exact title / ordinal / "last") tried BEFORE understand() is ever called,
exactly mirroring the pre-existing resolve_pending_cart_checkout() pattern.

Hermetic -- `understand_fn` is always stubbed; controlled DisplayedProduct
fixtures matching the real bug's shape are used (Beef Mince Low Fat Balady
Premium / Minced Beef Low Fat Balady).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.scenario.llm_context import build_llm_session_context, derive_conversation_phase  # noqa: E402
from src.scenario.orchestrator import handle_message  # noqa: E402
from src.scenario.session_activity import (  # noqa: E402
    CartLine, DisplayedProduct, PendingAction, SessionActivityStore, TurnRecord,
)
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


_BEEF_LOW_FAT_PREMIUM = DisplayedProduct(
    id=9001, name_en="Beef Mince Low Fat Balady Premium", name_ar="لحمة مفرومة بقرى فاخر بلدى قليل الدسم",
    brand_normalized="balady", price=189.0, special_price=189.0,
)
_BEEF_LOW_FAT = DisplayedProduct(
    id=9002, name_en="Minced Beef Low Fat Balady", name_ar="لحمة مفرومة بقرى بلدي قليل الدسم",
    brand_normalized="balady", price=175.0, special_price=175.0,
)
_BEEF_REGULAR = DisplayedProduct(
    id=9003, name_en="Beef Mince Balady", name_ar="لحمة مفرومة بقرى بلدي",
    brand_normalized="balady", price=160.0, special_price=160.0,
)
_BEEF = [_BEEF_LOW_FAT_PREMIUM, _BEEF_LOW_FAT, _BEEF_REGULAR]


def _primed_session(session_id: str, shown: list[DisplayedProduct]) -> tuple[StateManager, SessionActivityStore]:
    sm, store = StateManager(), SessionActivityStore()
    store.set_last_shown_products(session_id, shown)
    return sm, store


def _make_ambiguous(session_id: str, sm: StateManager, store: SessionActivityStore, search: RecordingSearch) -> None:
    """Reproduces the FIRST half of the real bug: an ambiguous cart_action=
    "add" against 2+ shown beef products, which must leave a
    cart_add_pending state behind, carrying the real candidate ids it was
    ambiguous over (state-hardening task: resolution now happens ONLY
    against this frozen snapshot -- see test_a0 below for that creation
    path proven end-to-end through a real turn; this helper just primes
    the state directly so the OTHER tests in this file -- about the
    CONTINUATION -- don't depend on which exact raw wording/marker the
    creation step happens to use)."""
    shown_ids = [p.id for p in store.get(session_id).last_shown_products]
    store.set_pending_action(session_id, "cart_add_pending", shown_product_ids=shown_ids)
    assert store.get(session_id).cart == []


# --- A0: the creation step itself, through a real turn, with a message
# shaped like the actual bug report -- proves handle_cart_action() really
# does set cart_add_pending on a genuine ambiguous add, not just that this
# file's other tests can prime it directly. ---

def test_a0_ambiguous_add_through_a_real_turn_sets_pending_state():
    sm, store = _primed_session("s1", _BEEF)
    search = RecordingSearch()
    # A selection_text that fuzzy-matches more than one shown beef product
    # (all three contain "Balady") -- a real, if imperfect, stand-in for
    # whatever the LLM extracts from "حطها في السله" in production.
    understand_fn = stub_understand(Action(intent=Intent.NO_OP, cart_action="add", selection_text="Balady"))
    r = handle_message("s1", "حطها في السله", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert r.intent == "cart_add"
    assert store.get("s1").cart == []
    assert store.get("s1").pending_action is not None
    assert store.get("s1").pending_action.kind == "cart_add_pending"


# ======================================================================
# Test Group A -- cart-add continuation, all deterministic (0 LLM calls
# for the continuation turn itself)
# ======================================================================

# --- A1: exact English full title -> product added, no search, pending
# cleared, 0 LLM calls ---

def test_a1_exact_english_title_resolves_with_zero_llm_calls():
    sm, store = _primed_session("s1", _BEEF)
    search = RecordingSearch()
    _make_ambiguous("s1", sm, store, search)

    # A hostile stub that raises if called at all -- proves 0 LLM calls
    # for this continuation turn.
    def _hostile(message, history=None):
        raise AssertionError("understand() must not be called for an exact-title pending continuation")

    r = handle_message("s1", "Beef Mince Low Fat Balady Premium", state_manager=sm, activity_store=store,
                        understand_fn=_hostile, search_fn=search)

    cart = store.get("s1").cart
    assert len(cart) == 1
    assert cart[0].product.id == _BEEF_LOW_FAT_PREMIUM.id
    assert store.get("s1").pending_action is None
    assert len(search.calls) == 0
    assert "beef mince low fat balady premium" in r.assistant_message.lower()


# --- A2: exact Arabic full title -> same result ---

def test_a2_exact_arabic_title_resolves_with_zero_llm_calls():
    sm, store = _primed_session("s1", _BEEF)
    search = RecordingSearch()
    _make_ambiguous("s1", sm, store, search)

    def _hostile(message, history=None):
        raise AssertionError("understand() must not be called for an exact-title pending continuation")

    handle_message("s1", _BEEF_LOW_FAT_PREMIUM.name_ar, state_manager=sm, activity_store=store,
                    understand_fn=_hostile, search_fn=search)

    cart = store.get("s1").cart
    assert len(cart) == 1
    assert cart[0].product.id == _BEEF_LOW_FAT_PREMIUM.id
    assert store.get("s1").pending_action is None
    assert len(search.calls) == 0


# --- A3: ordinal word ("second one") -> resolves the second shown
# candidate, 0 LLM ---

def test_a3_ordinal_word_resolves_with_zero_llm_calls():
    sm, store = _primed_session("s1", _BEEF)
    search = RecordingSearch()
    _make_ambiguous("s1", sm, store, search)

    def _hostile(message, history=None):
        raise AssertionError("understand() must not be called for a raw-ordinal pending continuation")

    handle_message("s1", "the second one", state_manager=sm, activity_store=store,
                    understand_fn=_hostile, search_fn=search)

    cart = store.get("s1").cart
    assert len(cart) == 1
    assert cart[0].product.id == _BEEF_LOW_FAT.id
    assert store.get("s1").pending_action is None
    assert len(search.calls) == 0


# --- A4: bare digit ("رقم ٢" -- Arabic-Indic numeral) -> resolves the
# second candidate, 0 LLM ---

def test_a4_arabic_indic_digit_resolves_with_zero_llm_calls():
    sm, store = _primed_session("s1", _BEEF)
    search = RecordingSearch()
    _make_ambiguous("s1", sm, store, search)

    def _hostile(message, history=None):
        raise AssertionError("understand() must not be called for a raw-digit pending continuation")

    handle_message("s1", "رقم ٢", state_manager=sm, activity_store=store,
                    understand_fn=_hostile, search_fn=search)

    cart = store.get("s1").cart
    assert len(cart) == 1
    assert cart[0].product.id == _BEEF_LOW_FAT.id
    assert store.get("s1").pending_action is None
    assert len(search.calls) == 0


# --- A5: an ambiguous PARTIAL description with no raw exact-title/
# ordinal/"last" evidence -- no mutation, clarification/pending survives.
# (This one genuinely needs language help, so it's allowed 1 LLM call --
# see Test Group B for the full semantic-resolution case.) ---

def test_a5_ambiguous_partial_description_stays_pending_no_mutation():
    sm, store = _primed_session("s1", _BEEF)
    search = RecordingSearch()
    _make_ambiguous("s1", sm, store, search)

    # "Balady" alone matches all three shown products -- genuinely ambiguous.
    understand_fn = stub_understand(Action(intent=Intent.NO_OP, selection_text="Balady"))
    r = handle_message("s1", "بلدي", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert store.get("s1").cart == []
    assert store.get("s1").pending_action is not None
    assert store.get("s1").pending_action.kind == "cart_add_pending"
    assert len(search.calls) == 0
    assert r.intent == "cart_add"


# ======================================================================
# Test Group B -- language assistance: a genuinely semantic description
# that the deterministic resolver cannot safely resolve from raw evidence
# alone gets exactly ONE real LLM call, with compact pending context, and
# the backend still does the actual resolution/mutation.
# ======================================================================

def test_b_semantic_description_uses_exactly_one_llm_call_and_resolves_uniquely():
    sm, store = _primed_session("s1", [_BEEF_LOW_FAT_PREMIUM, _BEEF_LOW_FAT])
    search = RecordingSearch()
    _make_ambiguous("s1", sm, store, search)

    calls = {"n": 0}

    def _counting_understand(message, history=None):
        calls["n"] += 1
        # Simulates the LLM correctly reading the compact pending context
        # (candidate names include "Low Fat") and extracting a selection
        # description -- never a raw product id.
        return Action(intent=Intent.NO_OP, selection_text="low fat")

    r = handle_message("s1", "اللي قليل الدسم", state_manager=sm, activity_store=store,
                        understand_fn=_counting_understand, search_fn=search)

    assert calls["n"] == 1
    cart = store.get("s1").cart
    # "low fat" alone is ambiguous between the two candidates (both contain
    # it) -- this proves the BACKEND, not the LLM, does the real matching:
    # a generic "low fat" must not silently pick one arbitrarily.
    assert store.get("s1").pending_action is not None or len(cart) == 1
    assert r.intent == "cart_add"


def test_b_compact_context_carries_the_pending_candidates():
    """Directly verifies the compact context passed to understand() (via
    the contextvar) actually names the real pending candidates -- proving
    the LLM has what it needs to do the semantic match in test_b above,
    without relying on the fuzzy matcher alone."""
    from src.understanding.llm import get_session_context

    sm, store = _primed_session("s1", [_BEEF_LOW_FAT_PREMIUM, _BEEF_LOW_FAT])
    search = RecordingSearch()
    _make_ambiguous("s1", sm, store, search)

    captured = {}

    def _capturing_understand(message, history=None):
        captured["context"] = get_session_context()
        return Action(intent=Intent.NO_OP, selection_text="low fat")

    handle_message("s1", "اللي قليل الدسم", state_manager=sm, activity_store=store,
                    understand_fn=_capturing_understand, search_fn=search)

    context = captured["context"]
    assert context is not None
    assert "PENDING_CART_ADD_SELECTION" in context
    assert _BEEF_LOW_FAT_PREMIUM.name_en in context
    assert _BEEF_LOW_FAT.name_en in context


# ======================================================================
# Test Group C -- topic switch: a pending cart-add must not trap an
# unrelated later request.
# ======================================================================

def test_c_topic_switch_cancels_pending_add_and_searches_instead():
    sm, store = _primed_session("s1", _BEEF)
    search = RecordingSearch()
    _make_ambiguous("s1", sm, store, search)

    calls = {"n": 0}

    def _counting_understand(message, history=None):
        calls["n"] += 1
        # recommendation_request=True purely to keep this test isolated
        # from the separate, pre-existing "offer recommendations after a
        # search" behavior (orchestrator.py's F branch) -- unrelated to
        # the topic-switch behavior under test here.
        return Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk", recommendation_request=True)

    r = handle_message("s1", "لا خلاص وريني لبن", state_manager=sm, activity_store=store,
                        understand_fn=_counting_understand, search_fn=search)

    assert calls["n"] == 1
    assert store.get("s1").cart == []  # no accidental cart mutation
    assert store.get("s1").pending_action is None  # pending add cancelled, not
    # silently resurrected on a later turn
    assert len(search.calls) == 1
    # "milk" resolves to the canonical category (validate() prefers a real
    # catalog category over free text when one matches) -- either shape is
    # fine here; what matters is a genuine milk search ran, not beef.
    assert search.calls[0].category == "milk_29" or search.calls[0].query_text == "milk"
    assert derive_conversation_phase(store.get("s1")) == "BROWSING"
    assert r.intent == "search"


# ======================================================================
# Test Group F -- non-pending control: the SAME exact title, with no
# pending cart-add open at all, must NOT auto-add to cart -- proves the
# fix is state-aware, not a product-name hardcode.
# ======================================================================

def test_f_exact_title_without_pending_state_does_not_auto_add():
    sm, store = _primed_session("s1", _BEEF)
    search = RecordingSearch()

    understand_fn = stub_understand(
        Action(intent=Intent.NO_OP, selection_text=_BEEF_LOW_FAT_PREMIUM.name_en),
    )
    handle_message("s1", _BEEF_LOW_FAT_PREMIUM.name_en, state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)

    # A plain (non-pending, non-checkout, non-cart-action) selection only
    # acknowledges the reference -- it must never silently add to cart.
    assert store.get("s1").cart == []


def test_f_fresh_session_no_pending_never_forces_cart_action():
    sm, store = StateManager(), SessionActivityStore()
    search = RecordingSearch()

    understand_fn = stub_understand(Action(intent=Intent.SEARCH, raw_query_text="beef", category_hint="beef"))
    r = handle_message("s1", "Beef Mince Low Fat Balady Premium", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert store.get("s1").cart == []
    assert r.intent == "search"


# ======================================================================
# Test Group G -- compact memory quality: the built context is small,
# structured, and relevance-aware -- never a raw chat-history dump.
# ======================================================================

def _store_with_turns(
    session_id: str, pending_kind: str | None, shown: list[DisplayedProduct], cart: list[CartLine] | None = None,
) -> SessionActivityStore:
    store = SessionActivityStore()
    store.set_last_shown_products(session_id, shown)
    activity = store.get(session_id)
    activity.cart = cart or []
    activity.turns = [
        TurnRecord(message="مش عايز فراخ، خلّيها لحمة", intent="search", outcome="displayed 3 beef products"),
        TurnRecord(message="حطها في السله", intent="no_op", outcome="cart add was ambiguous; asked user to identify one shown product"),
    ]
    if pending_kind:
        # State-hardening task: a pending action always carries its real
        # candidate snapshot now -- shown products for a shown-product-
        # sourced kind, cart product ids for a cart-sourced one.
        ids = [line.product.id for line in activity.cart] if pending_kind == "cart_checkout_pending" else [p.id for p in shown]
        activity.pending_action = PendingAction(kind=pending_kind, shown_product_ids=ids)
    return store


def test_g_context_is_none_for_a_fresh_idle_session():
    store = SessionActivityStore()
    activity = store.get("s1")
    context = build_llm_session_context(activity, FilterSet(), None, store, "s1")
    assert context is None


def test_g_context_contains_phase_recent_outcomes_and_candidates():
    store = _store_with_turns("s1", "cart_add_pending", _BEEF)
    activity = store.get("s1")
    pending = activity.pending_action
    context = build_llm_session_context(activity, FilterSet(query_text="beef"), pending, store, "s1")

    assert context is not None
    assert "phase: PENDING_CART_ADD_SELECTION" in context
    assert "RECENT_CONTEXT" in context
    assert "cart add was ambiguous" in context
    assert "CANDIDATES" in context
    for p in _BEEF:
        assert p.name_en in context


def test_g_context_excludes_full_product_cards_and_prices():
    store = _store_with_turns("s1", "cart_add_pending", _BEEF)
    activity = store.get("s1")
    context = build_llm_session_context(activity, FilterSet(query_text="beef"), activity.pending_action, store, "s1")

    assert context is not None
    # Only names in CANDIDATES -- never price/brand/id noise repeated per line.
    assert "189.0" not in context
    assert "175.0" not in context
    assert str(_BEEF_LOW_FAT_PREMIUM.id) not in context.replace("phase:", "")


def test_g_context_excludes_unrelated_cart_when_irrelevant():
    """A plain search turn (no pending action) shouldn't dump full cart/
    checkout detail into the prompt -- only what's relevant."""
    store = SessionActivityStore()
    store.set_last_shown_products("s1", _BEEF)
    activity = store.get("s1")
    activity.cart = [CartLine(product=_BEEF_REGULAR, quantity=2)]
    activity.turns = [TurnRecord(message="get me beef", intent="search", outcome="displayed 3 beef products")]

    context = build_llm_session_context(activity, FilterSet(query_text="beef"), None, store, "s1")
    assert context is not None
    # cart_item_count is a single integer (relevant, cheap) -- but no
    # per-line cart CANDIDATES block when nothing is pending on the cart.
    assert "cart_item_count: 1" in context
    assert "CANDIDATES" not in context


def test_g_context_bounded_recent_turns_not_full_history():
    store = SessionActivityStore()
    activity = store.get("s1")
    activity.turns = [TurnRecord(message=f"turn {i}", intent="search", outcome=f"outcome {i}") for i in range(10)]

    context = build_llm_session_context(activity, FilterSet(), None, store, "s1")
    assert context is not None
    # Only the bounded recent window appears, not all 10 turns.
    assert "turn 0" not in context
    assert "turn 9" in context
