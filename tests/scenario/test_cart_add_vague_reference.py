"""Tiny targeted fix: cart_action="add" with a vague SINGULAR reference
("it"/"this"/"ده"/"دي") and no other product evidence used to fall through
to the generic "not one of the products I showed you" wording -- wrong,
since the user IS trying to add something, just vaguely. With 2+ shown
products this is a genuine ambiguity (deterministic clarification, no
LLM call, no product added); with exactly one shown product it trivially
resolves (no other possible answer) and the add proceeds normally. See
src/scenario/orchestrator.py's _raw_message_has_vague_singular_reference()
and the cart_action == "add" branch.

Hermetic -- `understand_fn` is always stubbed; controlled DisplayedProduct
fixtures are used (same pattern as the other cart/checkout scenario
tests). Named/full-title cart-add and checkout are untouched by this fix
and are not re-tested here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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


_PEPSI_970 = DisplayedProduct(id=5293, name_en="Pepsi - 970 Ml", name_ar="بيبسي - 970 مل",
                               brand_normalized="pepsi", price=21.75, special_price=21.75)
_PEPSI_400 = DisplayedProduct(id=12939, name_en="Pepsi Carbonated Drink - 400Ml", name_ar="بيبسى مشروب غازى",
                               brand_normalized="pepsi", price=10.0, special_price=10.0)


def _primed_session(session_id: str, shown: list[DisplayedProduct]) -> tuple[StateManager, SessionActivityStore]:
    sm, store = StateManager(), SessionActivityStore()
    store.set_last_shown_products(session_id, shown)
    return sm, store


# --- A: vague "it" with 2+ shown products -> deterministic clarification,
# never the generic "not one of the products I showed you" wording, and
# nothing is added to the cart ---

def test_vague_it_with_multiple_shown_products_asks_which_one():
    sm, store = _primed_session("s1", [_PEPSI_970, _PEPSI_400])
    understand_fn = stub_understand(Action(intent=Intent.NO_OP, cart_action="add"))
    search = RecordingSearch()

    r = handle_message("s1", "i want to add it to cart please", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert r.intent == "cart_add"
    assert "which product" in r.assistant_message.lower()
    assert "wasn't one of the products" not in r.assistant_message.lower()
    assert store.get("s1").cart == []
    assert len(search.calls) == 0


# --- B: vague "this" with 2+ shown products -> same clarification ---

def test_vague_this_with_multiple_shown_products_asks_which_one():
    sm, store = _primed_session("s1", [_PEPSI_970, _PEPSI_400])
    understand_fn = stub_understand(Action(intent=Intent.NO_OP, cart_action="add"))
    search = RecordingSearch()

    r = handle_message("s1", "please add this to my cart", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert "which product" in r.assistant_message.lower()
    assert store.get("s1").cart == []


# --- C: vague Arabic "ده" with 2+ shown products -> Arabic clarification ---

def test_vague_arabic_reference_with_multiple_shown_products_asks_which_one():
    sm, store = _primed_session("s1", [_PEPSI_970, _PEPSI_400])
    understand_fn = stub_understand(Action(intent=Intent.NO_OP, cart_action="add"))
    search = RecordingSearch()

    r = handle_message("s1", "ضيف ده للسلة", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert "أنهي منتج" in r.assistant_message
    assert store.get("s1").cart == []


# --- D: vague "it" with exactly ONE shown product -> trivially resolves
# and the add proceeds normally ---

def test_vague_it_with_single_shown_product_adds_it():
    sm, store = _primed_session("s1", [_PEPSI_970])
    understand_fn = stub_understand(Action(intent=Intent.NO_OP, cart_action="add"))
    search = RecordingSearch()

    r = handle_message("s1", "i want to add it to cart please", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert r.intent == "cart_add"
    cart = store.get("s1").cart
    assert len(cart) == 1
    assert cart[0].product.id == _PEPSI_970.id
    assert cart[0].quantity == 1
    assert len(search.calls) == 0


# --- E: regression -- a genuine unmatched/unrelated reference (no vague
# marker word at all) still gets the ordinary "not found" wording ---

def test_unrelated_reference_still_gets_ordinary_not_found_wording():
    sm, store = _primed_session("s1", [_PEPSI_970, _PEPSI_400])
    understand_fn = stub_understand(Action(intent=Intent.NO_OP, cart_action="add", selection_text="xyzzyplugh"))
    search = RecordingSearch()

    r = handle_message("s1", "add the xyzzyplugh to my cart", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert "wasn't one of the products" in r.assistant_message.lower()
    assert store.get("s1").cart == []
