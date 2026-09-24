"""Sprint 6 checkout-submission task: structured in-chat DEMO order
submission + JSON persistence (src/scenario/checkout_submission.py,
src/checkout/demo_order.py). Hermetic -- `understand_fn` is always
stubbed for the setup turns (search/select/checkout-start); submission
itself calls submit_checkout() directly, never handle_message(), and
touches no LLM/search/recommendation/embedding path at all.

All tests use a TEMPORARY orders directory (pytest's tmp_path) -- never
the real project data/orders/.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.checkout.demo_order import SCHEMA_PATH  # noqa: E402
from src.scenario.checkout_submission import CheckoutSubmissionError, submit_checkout  # noqa: E402
from src.scenario.orchestrator import handle_message  # noqa: E402
from src.scenario.session_activity import CartLine, SessionActivityStore  # noqa: E402
from src.search_adapter.adapter import FilterSet, SearchAdapter, SearchResult  # noqa: E402
from src.state_manager.state_manager import StateManager  # noqa: E402
from src.understanding.schema import Action, Intent  # noqa: E402

import jsonschema  # noqa: E402


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


def _ready_checkout_session(session_id: str = "s1"):
    """Search -> select first product + start checkout in the SAME turn.
    Returns (state_manager, activity_store, snapshot, search).

    Cart/direct-checkout architecture task: bare "checkout" is now always
    CART-STATE-driven (never a reuse of an earlier, separate plain
    selection turn -- see orchestrator.py's C4). With no cart involved
    here, selection and checkout_request must arrive in the SAME turn.
    `snapshot` is the real DisplayedProduct (selected_products[0].product,
    unwrapped from its CartLine) so every existing caller's snapshot.id/
    name_en/price/... access keeps working unchanged."""
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"),
        Action(intent=Intent.NO_OP, selection_ordinals=[1], checkout_request=True),
    )
    search = RecordingSearch()
    handle_message(session_id, "show me milk", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    handle_message(session_id, "I'll take the first one and checkout", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    snapshot = store.get(session_id).selected_products[0].product
    assert store.get(session_id).checkout_active is True
    return sm, store, snapshot, search


VALID_FIELDS = dict(
    customer_name="Ahmed Mostafa",
    phone="01012345678",
    delivery_address="12 Tahrir Street, Cairo",
)


def _lines(snapshot, quantity: int = 2) -> list[tuple[int, int]]:
    return [(snapshot.id, quantity)]


def _ready_multi_checkout_session(session_id: str = "s1"):
    """Search -> add TWO shown products to cart -> checkout all. Returns
    (state_manager, activity_store, [snapshot_a, snapshot_b], search).

    Cart/direct-checkout architecture task: a raw message naming 2+ shown
    products is NEVER a direct multi-line checkout anymore -- a multi-
    product purchase requires the cart. This helper goes through the real
    cart flow instead: add both named products to cart (all-or-nothing,
    reusing the exact raw-message-authority multi-item matcher), then
    "checkout all" opens the cart-subset (here, cart-all) checkout draft.
    """
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="pepsi", brand_text="pepsi"),
        Action(intent=Intent.NO_OP, cart_action="add", selection_text="pepsi"),
        Action(intent=Intent.NO_OP, checkout_request=True),
    )
    search = RecordingSearch()
    handle_message(session_id, "pepsi", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    shown = store.get(session_id).last_shown_products[:2]
    assert len(shown) == 2
    raw = f"add {shown[0].name_en} and {shown[1].name_en} to cart"
    handle_message(session_id, raw, state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    assert len(store.get(session_id).cart) == 2  # sanity: both really landed in the cart
    handle_message(session_id, "checkout all", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    assert store.get(session_id).checkout_active is True
    return sm, store, shown, search


# --- A: valid submission saves exactly one file, matching the real product ---

def test_valid_submission_saves_one_file_for_the_real_product(tmp_path):
    sm, store, snapshot, search = _ready_checkout_session()
    orders_dir = tmp_path / "orders"

    result = submit_checkout(
        "s1", state_manager=sm, activity_store=store, orders_dir=orders_dir,
        lines=_lines(snapshot), **VALID_FIELDS,
    )

    files = list(orders_dir.glob("*.json"))
    assert len(files) == 1
    assert result["order"]["lines"][0]["id"] == snapshot.id


# --- B: saved JSON validates against the shared JSON Schema ---

def test_saved_order_validates_against_json_schema(tmp_path):
    sm, store, snapshot, search = _ready_checkout_session()
    orders_dir = tmp_path / "orders"
    submit_checkout("s1", state_manager=sm, activity_store=store, orders_dir=orders_dir, lines=_lines(snapshot), **VALID_FIELDS)

    saved = json.loads(next(orders_dir.glob("*.json")).read_text(encoding="utf-8"))
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(instance=saved, schema=schema)  # raises on failure
    assert saved["$schema"]


# --- C: correct real product identity in both file and response ---

def test_order_uses_the_real_selected_product_identity(tmp_path):
    sm, store, snapshot, search = _ready_checkout_session()
    orders_dir = tmp_path / "orders"
    result = submit_checkout("s1", state_manager=sm, activity_store=store, orders_dir=orders_dir, lines=_lines(snapshot), **VALID_FIELDS)

    saved = json.loads(next(orders_dir.glob("*.json")).read_text(encoding="utf-8"))
    assert saved["products"][0]["id"] == snapshot.id
    assert saved["products"][0]["name_en"] == snapshot.name_en
    assert saved["products"][0]["name_ar"] == snapshot.name_ar
    assert result["order"]["lines"][0]["name_en"] == snapshot.name_en


# --- D: payment method is always cash_on_delivery ---

def test_payment_method_always_cash_on_delivery(tmp_path):
    sm, store, snapshot, search = _ready_checkout_session()
    orders_dir = tmp_path / "orders"
    result = submit_checkout("s1", state_manager=sm, activity_store=store, orders_dir=orders_dir, lines=_lines(snapshot), **VALID_FIELDS)

    saved = json.loads(next(orders_dir.glob("*.json")).read_text(encoding="utf-8"))
    assert saved["payment_method"] == "cash_on_delivery"
    assert result["order"]["payment_method"] == "cash_on_delivery"


# --- E: client cannot override product/price/payment via extra kwargs (signature-level) ---

def test_submit_checkout_signature_has_no_client_controlled_identity_fields():
    import inspect
    sig = inspect.signature(submit_checkout)
    for forbidden in ("product_id", "product_name", "price", "payment_method", "order_id", "status"):
        assert forbidden not in sig.parameters


# --- F: special_price used when present, else regular price ---

def test_uses_special_price_when_present(tmp_path):
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"),
        Action(intent=Intent.NO_OP, selection_ordinals=[1], checkout_request=True),
    )
    search = RecordingSearch()
    handle_message("s1", "show me milk", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    shown = store.get("s1").last_shown_products
    # Force a known special_price on the cached snapshot to make the rule
    # deterministic regardless of real catalog data.
    shown[0].price = 100.0
    shown[0].special_price = 75.0
    store.set_last_shown_products("s1", shown)
    # Cart/direct-checkout architecture task: selection + checkout_request
    # combined in one turn (no cart involved) -- bare "checkout" is now
    # always cart-state-driven, never a reuse of an earlier plain selection.
    handle_message("s1", "I'll take the first one and checkout", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    snapshot = store.get("s1").selected_products[0].product

    orders_dir = tmp_path / "orders"
    result = submit_checkout("s1", state_manager=sm, activity_store=store, orders_dir=orders_dir, lines=_lines(snapshot), **VALID_FIELDS)
    assert result["order"]["lines"][0]["unit_price"] == 75.0


def test_uses_regular_price_when_no_special_price(tmp_path):
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"),
        Action(intent=Intent.NO_OP, selection_ordinals=[1], checkout_request=True),
    )
    search = RecordingSearch()
    handle_message("s1", "show me milk", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    shown = store.get("s1").last_shown_products
    shown[0].price = 42.5
    shown[0].special_price = None
    store.set_last_shown_products("s1", shown)
    handle_message("s1", "I'll take the first one and checkout", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    snapshot = store.get("s1").selected_products[0].product

    orders_dir = tmp_path / "orders"
    result = submit_checkout("s1", state_manager=sm, activity_store=store, orders_dir=orders_dir, lines=_lines(snapshot), **VALID_FIELDS)
    assert result["order"]["lines"][0]["unit_price"] == 42.5


def test_special_price_zero_sentinel_falls_back_to_regular_price(tmp_path):
    # Sprint 6 zero-price semantics task: real catalog convention --
    # special_price=0 is a "no special price recorded" sentinel (151 real
    # rows, e.g. "Kabochas" at price=9.95/special_price=0.00), never a
    # genuine free/zero sale price. Checkout must never charge a real
    # product at 0 just because special_price happened to be present.
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"),
        Action(intent=Intent.NO_OP, selection_ordinals=[1], checkout_request=True),
    )
    search = RecordingSearch()
    handle_message("s1", "show me milk", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    shown = store.get("s1").last_shown_products
    shown[0].price = 9.95
    shown[0].special_price = 0.0
    store.set_last_shown_products("s1", shown)
    handle_message("s1", "I'll take the first one and checkout", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    snapshot = store.get("s1").selected_products[0].product

    orders_dir = tmp_path / "orders"
    result = submit_checkout("s1", state_manager=sm, activity_store=store, orders_dir=orders_dir, lines=_lines(snapshot), **VALID_FIELDS)
    assert result["order"]["lines"][0]["unit_price"] == 9.95


def test_both_price_and_special_price_zero_is_rejected_not_fabricated(tmp_path):
    # A real (if unfortunate) ~78-row data gap: price itself is 0 with no
    # other authoritative field to recover a real number from. Checkout
    # must not invent a price. Unknown-price entry detection task: this is
    # now caught EARLIER, in orchestrator.py's C4 branch, before
    # checkout_active is ever set/a CheckoutForm is ever built -- not only
    # discovered later at submission. submit_checkout() itself keeps its
    # own independent guard too (demo_order.py's effective_price(), the
    # SAME rule, plus the unmodified JSON Schema's total_price
    # exclusiveMinimum 0) as defense in depth for direct/low-level calls.
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"),
        Action(intent=Intent.NO_OP, selection_ordinals=[1], checkout_request=True),
    )
    search = RecordingSearch()
    handle_message("s1", "show me milk", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    shown = store.get("s1").last_shown_products
    shown[0].price = 0.0
    shown[0].special_price = 0.0
    store.set_last_shown_products("s1", shown)
    # Cart/direct-checkout architecture task: selection + checkout_request
    # combined in one turn so the price guard is actually exercised on the
    # DIRECT checkout path (bare "checkout" after a separate plain
    # selection is cart-state-driven now, and no cart is involved here).
    r = handle_message("s1", "I'll take the first one and checkout", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert r.checkout_form is None
    assert store.get("s1").checkout_active is False
    assert "price" in r.assistant_message.lower()

    # Defense in depth: a direct/low-level submit_checkout() call (as if
    # checkout HAD somehow been opened) is still independently rejected.
    snapshot = shown[0]
    orders_dir = tmp_path / "orders"
    store.get("s1").checkout_active = True
    with pytest.raises(CheckoutSubmissionError):
        submit_checkout("s1", state_manager=sm, activity_store=store, orders_dir=orders_dir, lines=_lines(snapshot), **VALID_FIELDS)
    assert not orders_dir.exists() or list(orders_dir.glob("*.json")) == []


# --- G: total_price = unit_price * quantity ---

def test_total_price_is_unit_price_times_quantity(tmp_path):
    sm, store, snapshot, search = _ready_checkout_session()
    orders_dir = tmp_path / "orders"
    result = submit_checkout(
        "s1", state_manager=sm, activity_store=store, orders_dir=orders_dir,
        lines=_lines(snapshot, quantity=3), **VALID_FIELDS,
    )
    assert result["order"]["total_price"] == round(result["order"]["lines"][0]["unit_price"] * 3, 2)


# --- H: quantity <= 0 rejected, no file ---

@pytest.mark.parametrize("bad_qty", [0, -1, -5])
def test_non_positive_quantity_rejected(tmp_path, bad_qty):
    sm, store, snapshot, search = _ready_checkout_session()
    orders_dir = tmp_path / "orders"
    with pytest.raises(CheckoutSubmissionError):
        submit_checkout(
            "s1", state_manager=sm, activity_store=store, orders_dir=orders_dir,
            lines=_lines(snapshot, quantity=bad_qty), **VALID_FIELDS,
        )
    assert not orders_dir.exists() or list(orders_dir.glob("*.json")) == []
    assert store.get("s1").checkout_active is True  # never cleared on failure


# --- I: empty/whitespace customer_name rejected ---

@pytest.mark.parametrize("bad_name", ["", "   "])
def test_empty_customer_name_rejected(tmp_path, bad_name):
    sm, store, snapshot, search = _ready_checkout_session()
    orders_dir = tmp_path / "orders"
    fields = dict(VALID_FIELDS, customer_name=bad_name)
    with pytest.raises(CheckoutSubmissionError):
        submit_checkout("s1", state_manager=sm, activity_store=store, orders_dir=orders_dir, lines=_lines(snapshot), **fields)
    assert not orders_dir.exists() or list(orders_dir.glob("*.json")) == []
    assert store.get("s1").checkout_active is True


# --- J: valid Egyptian mobile phone formats accepted -- all four operator
# prefixes, both local (0...) and international (+20...) forms ---

@pytest.mark.parametrize("good_phone", [
    "01012345678", "01112345678", "01212345678", "01512345678",
    "+201012345678", "+201112345678", "+201212345678", "+201512345678",
])
def test_valid_egyptian_phone_formats_accepted(tmp_path, good_phone):
    sm, store, snapshot, search = _ready_checkout_session()
    orders_dir = tmp_path / "orders"
    fields = dict(VALID_FIELDS, phone=good_phone)
    result = submit_checkout(
        "s1", state_manager=sm, activity_store=store, orders_dir=orders_dir,
        lines=_lines(snapshot), **fields,
    )
    assert result["order"]["order_id"]


# --- K: malformed phone rejected -- missing leading 0, a stray 0 after
# "+20", a non-operator second digit, and wrong length, alongside the
# pre-existing generic-garbage cases ---

@pytest.mark.parametrize("bad_phone", [
    "123", "notaphone", "0199999", "0299123456789999",
    "1012345678",        # missing leading 0
    "+2001012345678",    # extra 0 after +20
    "01312345678",       # 3 is not a valid operator digit
    "+201312345678",     # same, international form
    "0101234567",        # too short (10 digits)
    "010123456789",      # too long (12 digits)
])
def test_malformed_phone_rejected(tmp_path, bad_phone):
    sm, store, snapshot, search = _ready_checkout_session()
    orders_dir = tmp_path / "orders"
    fields = dict(VALID_FIELDS, phone=bad_phone)
    with pytest.raises(CheckoutSubmissionError):
        submit_checkout("s1", state_manager=sm, activity_store=store, orders_dir=orders_dir, lines=_lines(snapshot), **fields)
    assert store.get("s1").checkout_active is True


# --- K: invalid/empty address rejected ---

@pytest.mark.parametrize("bad_address", ["", "  ", "abc"])
def test_invalid_address_rejected(tmp_path, bad_address):
    sm, store, snapshot, search = _ready_checkout_session()
    orders_dir = tmp_path / "orders"
    fields = dict(VALID_FIELDS, delivery_address=bad_address)
    with pytest.raises(CheckoutSubmissionError):
        submit_checkout("s1", state_manager=sm, activity_store=store, orders_dir=orders_dir, lines=_lines(snapshot), **fields)
    assert store.get("s1").checkout_active is True


# --- L: inactive checkout -> rejected, no file ---

def test_inactive_checkout_rejected(tmp_path):
    sm, store = StateManager(), SessionActivityStore()
    orders_dir = tmp_path / "orders"
    with pytest.raises(CheckoutSubmissionError):
        submit_checkout("s1", state_manager=sm, activity_store=store, orders_dir=orders_dir, lines=[(1, 1)], **VALID_FIELDS)
    assert not orders_dir.exists()


# --- M: missing/inconsistent snapshot -> rejected, no file ---

def test_missing_snapshot_rejected_even_if_checkout_active(tmp_path):
    sm, store = StateManager(), SessionActivityStore()
    store.get("s1").checkout_active = True  # active but no real selection -- inconsistent state
    orders_dir = tmp_path / "orders"
    with pytest.raises(CheckoutSubmissionError):
        submit_checkout("s1", state_manager=sm, activity_store=store, orders_dir=orders_dir, lines=[(1, 1)], **VALID_FIELDS)
    assert not orders_dir.exists()


def test_unknown_product_id_in_lines_rejected(tmp_path):
    sm, store, snapshot, search = _ready_checkout_session()
    orders_dir = tmp_path / "orders"
    with pytest.raises(CheckoutSubmissionError):
        submit_checkout(
            "s1", state_manager=sm, activity_store=store, orders_dir=orders_dir,
            lines=[(snapshot.id + 999999, 1)], **VALID_FIELDS,  # not the active selection
        )
    assert not orders_dir.exists()
    assert store.get("s1").checkout_active is True


# --- N: success -> checkout_active becomes False ---

def test_success_clears_checkout_active(tmp_path):
    sm, store, snapshot, search = _ready_checkout_session()
    orders_dir = tmp_path / "orders"
    submit_checkout("s1", state_manager=sm, activity_store=store, orders_dir=orders_dir, lines=_lines(snapshot), **VALID_FIELDS)
    assert store.get("s1").checkout_active is False


# --- O: duplicate immediate submission rejected, still exactly one file ---

def test_duplicate_submission_rejected_and_only_one_file_exists(tmp_path):
    sm, store, snapshot, search = _ready_checkout_session()
    orders_dir = tmp_path / "orders"
    submit_checkout("s1", state_manager=sm, activity_store=store, orders_dir=orders_dir, lines=_lines(snapshot), **VALID_FIELDS)
    assert len(list(orders_dir.glob("*.json"))) == 1

    with pytest.raises(CheckoutSubmissionError):
        submit_checkout("s1", state_manager=sm, activity_store=store, orders_dir=orders_dir, lines=_lines(snapshot), **VALID_FIELDS)

    assert len(list(orders_dir.glob("*.json"))) == 1


# --- P: JSON Schema validation failure -> no file, no fake success ---

def test_schema_validation_failure_leaves_no_file(tmp_path):
    sm, store, snapshot, search = _ready_checkout_session()
    orders_dir = tmp_path / "orders"
    bad_schema_path = tmp_path / "bad_schema.json"
    # A schema that no real order can ever satisfy (requires a field we
    # never produce) -- simulates a schema/order mismatch failure mode.
    bad_schema_path.write_text(json.dumps({
        "type": "object",
        "required": ["this_field_never_exists"],
    }), encoding="utf-8")

    with pytest.raises(CheckoutSubmissionError):
        submit_checkout(
            "s1", state_manager=sm, activity_store=store, lines=_lines(snapshot),
            orders_dir=orders_dir, schema_path=bad_schema_path, **VALID_FIELDS,
        )

    assert not orders_dir.exists() or list(orders_dir.glob("*.json")) == []
    assert store.get("s1").checkout_active is True  # not cleared -- persistence never succeeded
    # no leftover temp files either
    assert not orders_dir.exists() or list(orders_dir.glob("*.tmp")) == []


# --- Q: file-write failure -> no fake success, checkout remains recoverable ---

def test_write_failure_leaves_checkout_recoverable(tmp_path):
    sm, store, snapshot, search = _ready_checkout_session()
    # A regular FILE (not a directory) at the orders_dir path makes
    # orders_dir.mkdir()/tempfile.mkstemp() fail with a real OSError.
    blocked_path = tmp_path / "blocked_orders"
    blocked_path.write_text("not a directory", encoding="utf-8")

    with pytest.raises(CheckoutSubmissionError):
        submit_checkout("s1", state_manager=sm, activity_store=store, orders_dir=blocked_path, lines=_lines(snapshot), **VALID_FIELDS)

    assert store.get("s1").checkout_active is True


# --- R: atomic write -- no partial JSON ever visible under the final name ---

def test_atomic_write_leaves_no_temp_files_behind(tmp_path):
    sm, store, snapshot, search = _ready_checkout_session()
    orders_dir = tmp_path / "orders"
    submit_checkout("s1", state_manager=sm, activity_store=store, orders_dir=orders_dir, lines=_lines(snapshot), **VALID_FIELDS)

    all_files = list(orders_dir.iterdir())
    assert len(all_files) == 1
    assert all_files[0].suffix == ".json"
    # the file must be complete, valid JSON -- never a half-written artifact
    json.loads(all_files[0].read_text(encoding="utf-8"))


# --- S: order_id / filename are server-generated and path-safe ---

def test_order_id_is_server_generated_and_path_safe(tmp_path):
    sm, store, snapshot, search = _ready_checkout_session()
    orders_dir = tmp_path / "orders"
    result = submit_checkout("s1", state_manager=sm, activity_store=store, orders_dir=orders_dir, lines=_lines(snapshot), **VALID_FIELDS)

    order_id = result["order"]["order_id"]
    assert order_id
    assert "/" not in order_id and "\\" not in order_id and ".." not in order_id
    assert order_id not in VALID_FIELDS["customer_name"]
    assert order_id not in VALID_FIELDS["phone"]
    saved_path = orders_dir / f"{order_id}.json"
    assert saved_path.exists()


def test_order_id_not_derived_from_pii_even_with_pii_like_name(tmp_path):
    sm, store, snapshot, search = _ready_checkout_session()
    orders_dir = tmp_path / "orders"
    fields = dict(VALID_FIELDS, customer_name="01012345678")  # name that looks like the phone
    result = submit_checkout("s1", state_manager=sm, activity_store=store, orders_dir=orders_dir, lines=_lines(snapshot), **fields)
    assert result["order"]["order_id"] != "01012345678"


# --- T: session isolation ---

def test_submission_is_session_isolated(tmp_path):
    sm, store, snapshot, search = _ready_checkout_session("s1")
    orders_dir = tmp_path / "orders"

    with pytest.raises(CheckoutSubmissionError):
        submit_checkout("s2", state_manager=sm, activity_store=store, orders_dir=orders_dir, lines=_lines(snapshot), **VALID_FIELDS)

    assert not orders_dir.exists()
    assert store.get("s1").checkout_active is True  # s1's own state untouched


# --- U: normal conversational endpoint (handle_message) is unaffected ---

def test_normal_message_flow_unaffected_by_submission_module_import():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"))
    search = RecordingSearch()
    r = handle_message("s1", "I need milk", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)
    assert len(r.products) > 0
    assert r.checkout_form is None


# --- V: zero LLM/search/recommendation calls during submission itself ---

def test_submission_makes_no_llm_search_or_recommendation_calls(tmp_path):
    sm, store, snapshot, search = _ready_checkout_session()
    calls_before = len(search.calls)

    import src.scenario.checkout_submission as mod
    # submit_checkout has no understand_fn/search_fn parameter at all --
    # this just double-checks the module doesn't import/call them under
    # any alias.
    assert not hasattr(mod, "understand")
    assert not hasattr(mod, "SearchAdapter")
    assert not hasattr(mod, "RecommendationService")

    orders_dir = tmp_path / "orders"
    submit_checkout("s1", state_manager=sm, activity_store=store, orders_dir=orders_dir, lines=_lines(snapshot), **VALID_FIELDS)
    assert len(search.calls) == calls_before  # no new SearchAdapter call


# --- W: privacy -- PII never appears in the response payload ---

def test_response_never_echoes_customer_pii(tmp_path):
    sm, store, snapshot, search = _ready_checkout_session()
    orders_dir = tmp_path / "orders"
    result = submit_checkout("s1", state_manager=sm, activity_store=store, orders_dir=orders_dir, lines=_lines(snapshot), **VALID_FIELDS)

    dumped = json.dumps(result)
    assert VALID_FIELDS["customer_name"] not in dumped
    assert VALID_FIELDS["phone"] not in dumped
    assert VALID_FIELDS["delivery_address"] not in dumped
    assert "assistant_message" in result
    # confirmation wording must not claim real fulfillment
    msg = result["assistant_message"]
    for forbidden in ("payment received", "courier", "dispatched", "Spinneys has received"):
        assert forbidden.lower() not in msg.lower()


# ======================================================================
# Multi-product checkout submission (task: generalize existing checkout
# to support multiple shown products) -- two REAL shown products selected
# in one turn, one order, two lines.
# ======================================================================

# --- X: two exact shown products -> one order with 2 lines, correct
# per-line quantities and totals ---

def test_two_line_checkout_saves_one_order_with_both_lines(tmp_path):
    sm, store, snapshots, search = _ready_multi_checkout_session()
    orders_dir = tmp_path / "orders"
    lines = [(snapshots[0].id, 2), (snapshots[1].id, 5)]

    result = submit_checkout(
        "s1", state_manager=sm, activity_store=store, orders_dir=orders_dir,
        lines=lines, **VALID_FIELDS,
    )

    files = list(orders_dir.glob("*.json"))
    assert len(files) == 1  # ONE order, not two separate files
    order_lines = result["order"]["lines"]
    assert {line["id"] for line in order_lines} == {snapshots[0].id, snapshots[1].id}
    by_id = {line["id"]: line for line in order_lines}
    assert by_id[snapshots[0].id]["quantity"] == 2
    assert by_id[snapshots[1].id]["quantity"] == 5


# --- Y: authoritative line prices -- server computes unit_price/line_total
# from ITS OWN snapshot, quantity is the only client-influenced number ---

def test_multi_line_prices_are_server_authoritative_and_totals_correct(tmp_path):
    sm, store, snapshots, search = _ready_multi_checkout_session()
    orders_dir = tmp_path / "orders"
    lines = [(snapshots[0].id, 2), (snapshots[1].id, 3)]

    result = submit_checkout(
        "s1", state_manager=sm, activity_store=store, orders_dir=orders_dir,
        lines=lines, **VALID_FIELDS,
    )

    by_id = {line["id"]: line for line in result["order"]["lines"]}
    for snapshot, qty in zip(snapshots, (2, 3)):
        expected_unit = snapshot.special_price if snapshot.special_price and snapshot.special_price > 0 else snapshot.price
        assert by_id[snapshot.id]["unit_price"] == expected_unit
        assert by_id[snapshot.id]["line_total"] == round(expected_unit * qty, 2)
    assert result["order"]["total_price"] == round(
        sum(by_id[s.id]["line_total"] for s in snapshots), 2,
    )


# --- Z: client cannot submit a price/name -- CheckoutLineIn has no such field ---

def test_client_cannot_submit_product_price_or_name():
    import inspect
    from src.api.schemas import CheckoutLineIn
    fields = CheckoutLineIn.model_fields
    assert set(fields) == {"product_id", "quantity"}
    sig = inspect.signature(submit_checkout)
    for forbidden in ("price", "unit_price", "name_en", "name_ar"):
        assert forbidden not in sig.parameters


# --- AA: an unknown-price line is rejected safely, no partial order ---

def test_unknown_price_line_rejects_whole_multi_line_order(tmp_path):
    sm, store, snapshots, search = _ready_multi_checkout_session()
    # Force the second product's price to the real "no known price" gap.
    snapshots[1].price = None
    snapshots[1].special_price = None
    # selected_products is list[CartLine] now -- wrap the (still real,
    # already-selected) DisplayedProduct snapshots accordingly.
    store.set_selected_products("s1", [CartLine(product=p, quantity=1) for p in snapshots])

    orders_dir = tmp_path / "orders"
    lines = [(snapshots[0].id, 1), (snapshots[1].id, 1)]
    with pytest.raises(CheckoutSubmissionError):
        submit_checkout(
            "s1", state_manager=sm, activity_store=store, orders_dir=orders_dir,
            lines=lines, **VALID_FIELDS,
        )
    # No partial order -- neither line was persisted.
    assert not orders_dir.exists() or list(orders_dir.glob("*.json")) == []
    assert store.get("s1").checkout_active is True


# --- BB: submitting fewer lines than selected (dropping one) is rejected,
# never a silent partial checkout ---

def test_partial_lines_submission_rejected():
    sm, store, snapshots, search = _ready_multi_checkout_session()
    with pytest.raises(CheckoutSubmissionError):
        submit_checkout(
            "s1", state_manager=sm, activity_store=store,
            lines=[(snapshots[0].id, 1)],  # missing snapshots[1]
            **VALID_FIELDS,
        )
    assert store.get("s1").checkout_active is True


# --- CC: a line naming a product outside the active selection is rejected ---

def test_line_with_foreign_product_id_rejected():
    sm, store, snapshots, search = _ready_multi_checkout_session()
    with pytest.raises(CheckoutSubmissionError):
        submit_checkout(
            "s1", state_manager=sm, activity_store=store,
            lines=[(snapshots[0].id, 1), (999999, 1)],
            **VALID_FIELDS,
        )
    assert store.get("s1").checkout_active is True


# --- DD: two-line order validates against the shared JSON Schema ---

def test_two_line_order_validates_against_json_schema(tmp_path):
    sm, store, snapshots, search = _ready_multi_checkout_session()
    orders_dir = tmp_path / "orders"
    lines = [(snapshots[0].id, 1), (snapshots[1].id, 2)]
    submit_checkout(
        "s1", state_manager=sm, activity_store=store, orders_dir=orders_dir,
        lines=lines, **VALID_FIELDS,
    )

    saved = json.loads(next(orders_dir.glob("*.json")).read_text(encoding="utf-8"))
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(instance=saved, schema=schema)
    assert len(saved["products"]) == 2


# --- EE: single-product checkout regression -- unaffected by the
# multi-product generalization ---

def test_single_product_checkout_still_works_after_generalization(tmp_path):
    sm, store, snapshot, search = _ready_checkout_session()
    orders_dir = tmp_path / "orders"
    result = submit_checkout(
        "s1", state_manager=sm, activity_store=store, orders_dir=orders_dir,
        lines=_lines(snapshot, quantity=1), **VALID_FIELDS,
    )
    assert len(result["order"]["lines"]) == 1
    assert result["order"]["lines"][0]["id"] == snapshot.id


# ======================================================================
# Post-checkout continuation: a completed order must not poison the next
# shopping turn. Real browser symptom: "i want to buy another thing"
# right after a successful checkout got a generic "Sure, I found these
# products for you" with no real products -- the OLD canonical
# StateManager FilterSet (and the just-bought selected_products) survived
# submit_checkout() untouched, so the next vague turn inherited a
# "searchable target" that was really just the already-fulfilled order.
# ======================================================================

# --- FF: successful checkout clears BOTH the canonical shopping state
# and the selection -- not just checkout_active ---

def test_successful_checkout_clears_canonical_state_and_selection(tmp_path):
    sm, store, snapshot, search = _ready_checkout_session()
    orders_dir = tmp_path / "orders"
    submit_checkout(
        "s1", state_manager=sm, activity_store=store, orders_dir=orders_dir,
        lines=_lines(snapshot, quantity=1), **VALID_FIELDS,
    )

    assert store.get("s1").checkout_active is False
    assert store.get("s1").selected_products == []
    assert sm.get_state("s1") == FilterSet()


# --- GG: "i want to buy another thing" (no new product) after a
# successful checkout asks what's next -- no empty/stale search ---

def test_vague_continuation_after_checkout_asks_what_next_no_search(tmp_path):
    sm, store, snapshot, search = _ready_checkout_session()
    orders_dir = tmp_path / "orders"
    submit_checkout(
        "s1", state_manager=sm, activity_store=store, orders_dir=orders_dir,
        lines=_lines(snapshot, quantity=1), **VALID_FIELDS,
    )
    calls_before = len(search.calls)

    understand_fn = stub_understand(Action(intent=Intent.NO_OP))
    r = handle_message("s1", "i want to buy another thing", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert len(search.calls) == calls_before  # no empty catalog search
    assert r.products == []
    assert "tell me what product" in r.assistant_message.lower()


# --- HH: "i want another pepsi" after a successful checkout still runs a
# normal, correctly-scoped search -- the old order never interferes ---

def test_named_continuation_after_checkout_runs_normal_search(tmp_path):
    sm, store, snapshot, search = _ready_checkout_session()  # bought milk
    orders_dir = tmp_path / "orders"
    submit_checkout(
        "s1", state_manager=sm, activity_store=store, orders_dir=orders_dir,
        lines=_lines(snapshot, quantity=1), **VALID_FIELDS,
    )
    calls_before = len(search.calls)

    understand_fn = stub_understand(Action(intent=Intent.SEARCH, raw_query_text="pepsi", brand_text="pepsi"))
    r = handle_message("s1", "i want another pepsi", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert len(search.calls) == calls_before + 1
    assert r.resolved_brand == "pepsi"
    assert len(r.products) > 0


# --- II: a new checkout after a completed one still works normally ---

def test_new_checkout_after_completed_one_works_normally(tmp_path):
    sm, store, snapshot, search = _ready_checkout_session()
    orders_dir = tmp_path / "orders"
    submit_checkout(
        "s1", state_manager=sm, activity_store=store, orders_dir=orders_dir,
        lines=_lines(snapshot, quantity=1), **VALID_FIELDS,
    )

    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="pepsi", brand_text="pepsi"),
        Action(intent=Intent.NO_OP, selection_ordinals=[1], checkout_request=True),
    )
    handle_message("s1", "i want another pepsi", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    shown = store.get("s1").last_shown_products
    r = handle_message("s1", "i want to buy the first one", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert r.checkout_form is not None
    assert r.checkout_form.lines[0].id == shown[0].id
    assert store.get("s1").checkout_active is True

    result2 = submit_checkout(
        "s1", state_manager=sm, activity_store=store, orders_dir=orders_dir,
        lines=[(shown[0].id, 1)], **VALID_FIELDS,
    )
    assert result2["order"]["lines"][0]["id"] == shown[0].id
    assert store.get("s1").checkout_active is False


# ======================================================================
# Checkout confirmation language source: must come from the session's own
# conversational history (the SAME signal every other response already
# keys language off of via is_arabic_dominant()), never from
# customer-submitted PII (name/phone/address), which carries no signal
# about what language the CHAT itself is in.
# ======================================================================

def _checkout_session_with_conversation(messages: tuple[str, ...], selection_action: Action, session_id: str = "s1"):
    """Runs a real search -> combined select+checkout conversation using
    the given raw messages (oldest first), so activity.history ends with
    the LAST one -- the conversational language signal under test.

    Cart/direct-checkout architecture task: selection + checkout_request
    must arrive in the SAME turn (no cart involved here) -- `messages`
    is (search message, combined select+checkout message) and
    `selection_action` must already carry checkout_request=True alongside
    its selection field."""
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"),
        selection_action,
    )
    search = RecordingSearch()
    for message in messages:
        handle_message(session_id, message, state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)
    snapshot = store.get(session_id).selected_products[0].product
    return sm, store, snapshot


# --- JJ: English conversation + Arabic customer name/address -> English
# confirmation (PII must never flip it) ---

def test_english_conversation_with_arabic_pii_stays_english(tmp_path):
    sm, store, snapshot = _checkout_session_with_conversation(
        ("show me milk", "I'll take the first one and checkout"),
        Action(intent=Intent.NO_OP, selection_ordinals=[1], checkout_request=True),
    )
    orders_dir = tmp_path / "orders"
    fields = dict(VALID_FIELDS, customer_name="أحمد محمد", delivery_address="شارع التحرير، القاهرة")

    result = submit_checkout(
        "s1", state_manager=sm, activity_store=store, orders_dir=orders_dir,
        lines=_lines(snapshot), **fields,
    )

    assert "submitted successfully" in result["assistant_message"]


# --- KK: Arabic conversation + English customer name/address -> Arabic
# confirmation (PII must never flip it) ---

def test_arabic_conversation_with_english_pii_stays_arabic(tmp_path):
    sm, store, snapshot = _checkout_session_with_conversation(
        ("عايز حليب", "عايز اشتري الاخير"),
        Action(intent=Intent.NO_OP, selection_last=True, checkout_request=True),
    )
    orders_dir = tmp_path / "orders"
    fields = dict(VALID_FIELDS, customer_name="Ahmed Mohamed", delivery_address="123 Main Street, Cairo")

    result = submit_checkout(
        "s1", state_manager=sm, activity_store=store, orders_dir=orders_dir,
        lines=_lines(snapshot), **fields,
    )

    assert "تسجيل" in result["assistant_message"]
    assert "submitted successfully" not in result["assistant_message"]
