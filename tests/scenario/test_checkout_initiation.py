"""Sprint 6 checkout task: checkout intent + checkout-form contract from a
REAL previously-selected product only (src/scenario/orchestrator.py's C4
branch). Hermetic -- `understand_fn` is always stubbed; RecordingSearch
wraps the REAL SearchAdapter/DB so setup turns populate genuine
last_shown_products/selected_product context; RecommendationService is
never touched by anything here.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.scenario.orchestrator import handle_message  # noqa: E402
from src.scenario.session_activity import DisplayedProduct, SessionActivityStore  # noqa: E402
from src.search_adapter.adapter import FilterSet, SearchAdapter, SearchResult  # noqa: E402
from src.state_manager.state_manager import StateManager  # noqa: E402
from src.understanding.schema import Action, Intent  # noqa: E402

LOGGER_NAME = "src.scenario.orchestrator"


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


def _last_trace(caplog) -> dict:
    records = [r for r in caplog.records if r.name == LOGGER_NAME and r.msg == "request_trace %s"]
    assert records, "expected a request_trace log record"
    return records[-1].args


def _search_select_and_checkout(checkout_action: Action, caplog=None):
    # Cart/direct-checkout architecture task: bare "checkout" is now always
    # CART-STATE-driven (never a reuse of an earlier, separate plain
    # selection turn -- see orchestrator.py's C4). With no cart involved
    # here, selection and checkout_request must arrive in the SAME turn
    # (has_selection_reference + effective_checkout_request together) --
    # `checkout_action` is merged with selection_ordinals=[1] so a single
    # "...and checkout" turn resolves the DIRECT checkout in one step,
    # exactly like a real "I'll take the first one and checkout" message.
    sm, store = StateManager(), SessionActivityStore()
    combined_action = checkout_action.model_copy(update={"selection_ordinals": [1]})
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"),
        combined_action,
    )
    search = RecordingSearch()
    handle_message("s1", "show me milk", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    snapshot = store.get("s1").last_shown_products[0]

    if caplog is not None:
        with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
            r = handle_message("s1", "I'll take the first one and checkout", state_manager=sm, activity_store=store,
                                understand_fn=understand_fn, search_fn=search)
    else:
        r = handle_message("s1", "I'll take the first one and checkout", state_manager=sm, activity_store=store,
                            understand_fn=understand_fn, search_fn=search)
    return r, snapshot, store, search


# --- A: real selected product + checkout_request -> checkout_active, form ---

def test_checkout_starts_from_real_selected_product(caplog):
    r, snapshot, store, search = _search_select_and_checkout(
        Action(intent=Intent.NO_OP, checkout_request=True), caplog,
    )
    assert store.get("s1").checkout_active is True
    assert r.checkout_form is not None
    assert len(r.checkout_form.lines) == 1
    assert r.checkout_form.lines[0].id == snapshot.id
    assert r.checkout_form.lines[0].name_en == snapshot.name_en
    trace = _last_trace(caplog)
    assert trace["route"] == "checkout_start"
    assert trace["llm_calls"] == 1
    assert trace["search_calls"] == 0
    assert trace["recommendation_calls"] == 0
    assert trace["embedding_calls"] == 0


# --- B/C: bare checkout with nothing selected/no cart -> cart-state
# clarification, no guessing, no auto-select (cart/direct-checkout
# architecture task: bare "checkout" is CART-STATE-driven, never a reuse of
# an earlier plain selection -- with the cart empty here, this is the
# deterministic "cart is empty" wording, not the old checkout_no_selection_
# response wording, which no longer exists). ---

def test_no_selected_product_asks_which_one_and_does_not_guess():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"),
        Action(intent=Intent.NO_OP, checkout_request=True),
    )
    search = RecordingSearch()
    handle_message("s1", "show me milk", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    assert store.get("s1").selected_products == []  # sanity: nothing selected yet
    assert store.get("s1").cart == []  # sanity: nothing in the cart either

    r = handle_message("s1", "I want to checkout", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert r.checkout_form is None
    assert store.get("s1").checkout_active is False
    assert "cart is empty" in r.assistant_message.lower()


# --- D: checkout form fields ---

def test_checkout_form_has_required_fields():
    r, snapshot, store, search = _search_select_and_checkout(Action(intent=Intent.NO_OP, checkout_request=True))
    form = r.checkout_form
    assert hasattr(form.fields, "customer_name") and form.fields.customer_name is None
    assert hasattr(form.fields, "phone") and form.fields.phone is None
    assert hasattr(form.fields, "delivery_address") and form.fields.delivery_address is None
    assert form.payment_method == "cash_on_delivery"


# --- E: real price/name in the form ---

def test_checkout_form_uses_real_product_data():
    r, snapshot, store, search = _search_select_and_checkout(Action(intent=Intent.NO_OP, checkout_request=True))
    assert r.checkout_form.lines[0].price == snapshot.price
    assert r.checkout_form.lines[0].special_price == snapshot.special_price
    assert r.checkout_form.lines[0].name_ar == snapshot.name_ar


# --- F/G: no SearchAdapter, no RecommendationService for checkout itself ---

def test_checkout_initiation_makes_no_search_or_recommendation_call():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"),
        Action(intent=Intent.NO_OP, selection_ordinals=[1]),
        Action(intent=Intent.NO_OP, checkout_request=True),
    )
    search = RecordingSearch()
    handle_message("s1", "show me milk", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    handle_message("s1", "I want the first one", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    calls_before = len(search.calls)

    handle_message("s1", "checkout", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)

    assert len(search.calls) == calls_before  # no new SearchAdapter call


# --- H: recommendation pending + checkout request ---

def test_pending_recommendation_offer_does_not_swallow_checkout():
    # Cart/direct-checkout architecture task: selection + checkout_request
    # must arrive in the SAME turn here (no cart involved) -- this combined
    # turn is still the "very next turn" after the search that set the
    # pending offer, so it still exercises "does the pending offer swallow
    # this turn" exactly as before.
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"),
        Action(intent=Intent.NO_OP, selection_ordinals=[1], checkout_request=True),
    )
    search = RecordingSearch()
    handle_message("s1", "show me milk", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    assert store.get("s1").pending_action is not None  # the search created an offer

    r = handle_message("s1", "I'll take the first one and checkout", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert r.recommendations == []
    assert r.checkout_form is not None
    assert store.get("s1").pending_action is None


# --- I: RESET clears checkout_active ---

def test_reset_clears_checkout_active():
    # Cart/direct-checkout architecture task: selection + checkout_request
    # combined in one turn (no cart involved) -- see the module-level note
    # on _search_select_and_checkout().
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"),
        Action(intent=Intent.NO_OP, selection_ordinals=[1], checkout_request=True),
        Action(intent=Intent.RESET),
    )
    search = RecordingSearch()
    handle_message("s1", "show me milk", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    handle_message("s1", "I'll take the first one and checkout", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    assert store.get("s1").checkout_active is True

    handle_message("s1", "start over", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)

    assert store.get("s1").checkout_active is False


# --- J: a new normal shopping flow clears stale checkout state ---

def test_new_search_clears_stale_checkout_state():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"),
        Action(intent=Intent.NO_OP, selection_ordinals=[1], checkout_request=True),
        Action(intent=Intent.SEARCH, raw_query_text="cheese", category_hint="cheese"),
    )
    search = RecordingSearch()
    handle_message("s1", "show me milk", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    handle_message("s1", "I'll take the first one and checkout", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    assert store.get("s1").checkout_active is True

    handle_message("s1", "actually show me cheese", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)

    assert store.get("s1").checkout_active is False


# --- K: session isolation ---

def test_checkout_state_is_session_isolated():
    r, snapshot, store, search = _search_select_and_checkout(Action(intent=Intent.NO_OP, checkout_request=True))
    assert store.get("s1").checkout_active is True
    store2 = SessionActivityStore()
    assert store2.get("s2").checkout_active is False
    assert store2.get("s2").selected_products == []


# --- L: session expiry clears checkout state ---

def test_session_expiry_clears_checkout_state():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"),
        Action(intent=Intent.NO_OP, selection_ordinals=[1], checkout_request=True),
    )
    search = RecordingSearch()
    handle_message("s1", "show me milk", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search, now=0.0)
    handle_message("s1", "I'll take the first one and checkout", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search, now=1.0)
    assert store.get("s1", now=1.0).checkout_active is True

    from src.scenario.session_activity import SESSION_TTL_SECONDS
    understand_fn2 = stub_understand(Action(intent=Intent.OFF_TOPIC))
    handle_message("s1", "hello there", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn2, search_fn=search, now=SESSION_TTL_SECONDS + 100.0)

    assert store.get("s1", now=SESSION_TTL_SECONDS + 100.0).checkout_active is False
    assert store.get("s1", now=SESSION_TTL_SECONDS + 100.0).selected_products == []


# --- M: product-selection regression (pure selection, no checkout) ---

def test_pure_selection_without_checkout_still_works():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"),
        Action(intent=Intent.NO_OP, selection_ordinals=[1]),
    )
    search = RecordingSearch()
    handle_message("s1", "show me milk", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    shown = store.get("s1").last_shown_products
    r = handle_message("s1", "I want the first one", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert r.checkout_form is None
    assert store.get("s1").checkout_active is False
    # Cart/direct-checkout architecture task: a PLAIN selection acknowledges
    # without writing selected_products (reserved for an open checkout draft).
    assert store.get("s1").selected_products == []
    assert shown[0].name_en in r.assistant_message


# --- N: normal search/recommendation regression ---

def test_normal_search_has_null_checkout_form():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"))
    search = RecordingSearch()
    r = handle_message("s1", "I need milk", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)
    assert r.checkout_form is None
    assert len(r.products) > 0


# --- O: same-turn selection + checkout resolves the new selection first ---

def test_same_turn_selection_and_checkout():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"),
        Action(intent=Intent.NO_OP, selection_ordinals=[2], checkout_request=True),
    )
    search = RecordingSearch()
    handle_message("s1", "show me milk", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    shown = store.get("s1").last_shown_products

    r = handle_message("s1", "I'll take the second one and checkout", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert store.get("s1").selected_products[0].product.id == shown[1].id
    assert r.checkout_form is not None
    assert r.checkout_form.lines[0].id == shown[1].id


# ======================================================================
# Purchase-intent continuation ("yes i want to buy it" after a plain
# search, real browser report). A single live Groq call (history=["show
# me pepsi"], message="yes i want to buy it", selectable context on)
# confirmed the REAL Action shape: intent=NO_OP, selection_last=True,
# selection_text="Pepsi", checkout_request=True -- "it" resolved to
# selection_last=True, not a bare unresolved reference, so the EXISTING
# C3/C4 branch already honors checkout_request/selection_last regardless
# of action.intent, reaching zero new SearchAdapter calls either way.
#
# A LATER task found the real product-selection defect this exposed: with
# several real Pepsi products shown, "Pepsi" (the co-present selection_text)
# is itself only a bare, ambiguous brand -- the model had no actual
# positional evidence for "it" and silently defaulted to "last", which
# _resolve_product_selection() used to trust outright, wrongly checking
# out whichever product happened to be last. The resolver now trusts
# selection_last/selection_ordinal only when there's no co-present
# selection_text, or when it independently corroborates the SAME product
# -- so this exact shape now correctly asks for clarification instead.
# ======================================================================

def test_purchase_intent_continuation_after_search_with_ambiguous_text_asks_for_clarification(caplog):
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="pepsi", brand_text="pepsi"),
        Action(intent=Intent.NO_OP, selection_last=True, selection_text="Pepsi", checkout_request=True),
    )
    search = RecordingSearch()
    handle_message("s1", "show me pepsi", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    calls_before = len(search.calls)

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        r = handle_message("s1", "yes i want to buy it", state_manager=sm, activity_store=store,
                            understand_fn=understand_fn, search_fn=search)

    assert len(search.calls) == calls_before  # no repeated catalog search
    assert store.get("s1").selected_products == []  # never silently selected
    assert store.get("s1").checkout_active is False
    assert r.checkout_form is None
    assert "more than one match" in r.assistant_message.lower()
    assert store.get("s1").pending_action is not None
    assert store.get("s1").pending_action.kind == "checkout_pending_selection"
    trace = _last_trace(caplog)
    # Production fix: checkout_request=True makes this a failed CHECKOUT
    # attempt (not a plain selection), so trace.route is now the accurate
    # "checkout_clarification" -- previously this stayed mislabeled at
    # whatever the C3/C4 combined block set before the checkout-specific
    # failure path even ran.
    assert trace["route"] == "checkout_clarification"
    assert trace["search_calls"] == 0


def test_purchase_intent_continuation_with_single_shown_product():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.NO_OP, selection_ordinals=[1]),
        Action(intent=Intent.NO_OP, selection_last=True, selection_text="Pepsi", checkout_request=True),
    )
    search = RecordingSearch()
    store.set_last_shown_products("s1", [
        DisplayedProduct(
            id=42, name_en="Pepsi - Carbonated Drink - 1.5L", name_ar="بيبسي", brand_normalized="pepsi",
            price=24.95, special_price=24.95,
        ),
    ])
    handle_message("s1", "the first one", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    calls_before = len(search.calls)

    r = handle_message("s1", "yes i want to buy it", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert len(search.calls) == calls_before
    assert r.checkout_form is not None
    assert r.checkout_form.lines[0].id == 42


def test_unrelated_new_search_after_shown_products_still_searches():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="pepsi", brand_text="pepsi"),
        Action(intent=Intent.SEARCH, raw_query_text="chocolate", category_hint="chocolate"),
    )
    search = RecordingSearch()
    handle_message("s1", "show me pepsi", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    calls_before = len(search.calls)

    r = handle_message("s1", "actually show me chocolate", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert len(search.calls) == calls_before + 1  # a genuinely new search still runs
    assert r.resolved_category is not None
    assert r.checkout_form is None


# ======================================================================
# Unknown-price checkout entry: an unknown-price product (both price and
# special_price are 0 -- the catalog's own "no price recorded" shape,
# src/checkout/demo_order.py's effective_price()) must never reach a
# CheckoutForm -- caught here, before start_checkout()/CheckoutForm, not
# only later when submission fails against the JSON Schema.
# ======================================================================

_UNKNOWN_PRICE = DisplayedProduct(
    id=901, name_en="Mystery Item", name_ar="سلعة غامضة", brand_normalized="mystery",
    price=0.0, special_price=0.0,
)
_KNOWN_PRICE = DisplayedProduct(
    id=902, name_en="Known Item", name_ar="سلعة معروفة", brand_normalized="known",
    price=10.0, special_price=0.0,
)
_SPECIAL_PRICE = DisplayedProduct(
    id=903, name_en="Special Item", name_ar="سلعة مخفضة", brand_normalized="special",
    price=20.0, special_price=15.0,
)


def test_single_unknown_price_product_blocks_checkout_before_form():
    sm, store = StateManager(), SessionActivityStore()
    store.set_last_shown_products("s1", [_UNKNOWN_PRICE])
    understand_fn = stub_understand(Action(intent=Intent.NO_OP, selection_ordinals=[1], checkout_request=True))
    search = RecordingSearch()

    r = handle_message("s1", "buy the first one", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert r.checkout_form is None
    assert store.get("s1").checkout_active is False
    assert "price" in r.assistant_message.lower()
    assert "Mystery Item" in r.assistant_message


def test_multi_product_with_one_unknown_price_blocks_whole_checkout():
    sm, store = StateManager(), SessionActivityStore()
    store.set_last_shown_products("s1", [_KNOWN_PRICE, _UNKNOWN_PRICE])
    understand_fn = stub_understand(Action(intent=Intent.NO_OP, selection_text="item", checkout_request=True))
    search = RecordingSearch()

    raw = f"i want to buy {_KNOWN_PRICE.name_en} and {_UNKNOWN_PRICE.name_en}"
    r = handle_message("s1", raw, state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert r.checkout_form is None  # no partial checkout with just the known-price line
    assert store.get("s1").checkout_active is False
    assert "Mystery Item" in r.assistant_message


def test_price_greater_than_zero_special_zero_checkout_allowed():
    sm, store = StateManager(), SessionActivityStore()
    store.set_last_shown_products("s1", [_KNOWN_PRICE])
    understand_fn = stub_understand(Action(intent=Intent.NO_OP, selection_ordinals=[1], checkout_request=True))
    search = RecordingSearch()

    r = handle_message("s1", "buy the first one", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert r.checkout_form is not None
    assert r.checkout_form.lines[0].id == _KNOWN_PRICE.id


def test_special_price_positive_and_below_price_checkout_allowed():
    sm, store = StateManager(), SessionActivityStore()
    store.set_last_shown_products("s1", [_SPECIAL_PRICE])
    understand_fn = stub_understand(Action(intent=Intent.NO_OP, selection_ordinals=[1], checkout_request=True))
    search = RecordingSearch()

    r = handle_message("s1", "buy the first one", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert r.checkout_form is not None
    assert r.checkout_form.lines[0].id == _SPECIAL_PRICE.id
