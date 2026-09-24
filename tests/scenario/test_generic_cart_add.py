"""Generic cart-add correction: a bare/pronoun/plural cart add ("add to cart",
"add them", "حط في السلة", "حطهم في الكارت", ...) is a CART ACTION that names no
product -- it must never be resolved as a product name (the old symptom: the
command text was copied into selection_text, failed to match any shown
product, and produced "that wasn't one of the products I showed you").

The fix is structural, not lexical: the LLM classifies HOW the item is
referred to (Action.generic_reference), and the deterministic backend then
decides from state -- one shown product -> add it; several -> clarify with a
durable cart_add_pending; a plural reference -> the existing multi-product add;
mid-clarification -> the SAME frozen candidates. So these tests deliberately
vary the wording (it is only ever an input, never something the code matches)
and assert on state, not on phrases.

Hermetic -- `understand_fn` is always stubbed; controlled DisplayedProduct
fixtures, same pattern as test_cart_add_vague_reference.py.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.scenario.cart_flow import _names_a_product  # noqa: E402
from src.scenario.orchestrator import handle_message  # noqa: E402
from src.scenario.session_activity import DisplayedProduct, SessionActivityStore  # noqa: E402
from src.search_adapter.adapter import FilterSet, SearchResult  # noqa: E402
from src.state_manager.state_manager import StateManager  # noqa: E402
from src.understanding.schema import Action, Intent  # noqa: E402


def stub_understand(*actions: Action):
    it = iter(actions)

    def _stub(message: str, history: list[str] | None = None) -> Action:
        return next(it)

    return _stub


def no_search(filters: FilterSet, limit: int, offset: int) -> SearchResult:
    raise AssertionError("a cart add must never trigger a search")


def _p(pid: int, en: str, ar: str, price: float) -> DisplayedProduct:
    return DisplayedProduct(id=pid, name_en=en, name_ar=ar, brand_normalized=None, price=price, special_price=price)


_A = _p(101, "Alpha Crunchy Chips - 30G", "ألفا شيبسي - 30 جم", 10.0)
_B = _p(102, "Bravo Salted Chips - 45G", "برافو شيبسي - 45 جم", 12.0)
_C = _p(103, "Charlie Paprika Chips - 60G", "تشارلي شيبسي - 60 جم", 15.0)
_D = _p(104, "Delta Cheese Puffs - 25G", "دلتا فطائر - 25 جم", 8.0)
_E = _p(105, "Echo Corn Puffs - 40G", "إيكو ذرة - 40 جم", 9.0)


def _session(shown: list[DisplayedProduct], sid: str = "s1") -> tuple[StateManager, SessionActivityStore]:
    sm, store = StateManager(), SessionActivityStore()
    store.set_last_shown_products(sid, shown)
    return sm, store


def _send(sm, store, message, action, sid="s1"):
    return handle_message(sid, message, state_manager=sm, activity_store=store,
                          understand_fn=stub_understand(action), search_fn=no_search)


def _cart_ids(store, sid="s1") -> list[int]:
    return [line.product.id for line in store.get(sid).cart]


# Wording varies on purpose; the assertions never look at it.
_GENERIC_WORDINGS = [
    "add to cart",
    "i want to add to cart",
    "حط في السلة",
    "حطهم في الكارت",
    "ضيفهم للسلة",
]


# --- A: the command text the LLM may copy into selection_text is never a
# product name. generic_reference is what makes it generic. ---

@pytest.mark.parametrize("wording", _GENERIC_WORDINGS)
def test_generic_add_with_several_shown_products_clarifies_with_pending(wording):
    sm, store = _session([_A, _B, _C])
    action = Action(intent=Intent.NO_OP, cart_action="add", generic_reference="singular", selection_text=wording)

    r = _send(sm, store, wording, action)

    assert r.intent == "cart_add"
    assert "wasn't one of the products" not in r.assistant_message.lower()
    assert "ده مش من المنتجات" not in r.assistant_message
    assert _cart_ids(store) == []
    pending = store.get("s1").pending_action
    assert pending is not None and pending.kind == "cart_add_pending"
    assert set(pending.shown_product_ids) == {_A.id, _B.id, _C.id}


@pytest.mark.parametrize("wording", _GENERIC_WORDINGS)
def test_generic_add_with_one_shown_product_adds_it(wording):
    sm, store = _session([_A])
    action = Action(intent=Intent.NO_OP, cart_action="add", generic_reference="singular", selection_text=wording)

    _send(sm, store, wording, action)

    assert _cart_ids(store) == [_A.id]
    assert store.get("s1").pending_action is None


def test_bare_cart_add_with_no_selection_fields_is_generic_without_any_marker_word():
    """No generic_reference and no selection_* at all: nothing was named or
    selected, so it is generic by construction -- no pronoun list involved."""
    sm, store = _session([_A, _B])
    action = Action(intent=Intent.NO_OP, cart_action="add")

    r = _send(sm, store, "put the thing in my basket", action)

    assert "wasn't one of the products" not in r.assistant_message.lower()
    assert _cart_ids(store) == []
    assert store.get("s1").pending_action.kind == "cart_add_pending"


# --- B: a clear plural reference -> the existing multi-product add ---

@pytest.mark.parametrize("wording", ["add them to cart", "حطهم في الكارت", "put all of them in the cart"])
def test_plural_generic_add_adds_the_whole_current_shown_group(wording):
    sm, store = _session([_A, _B, _C])
    action = Action(intent=Intent.NO_OP, cart_action="add", generic_reference="plural", selection_text=wording)

    r = _send(sm, store, wording, action)

    assert _cart_ids(store) == [_A.id, _B.id, _C.id]
    assert store.get("s1").pending_action is None
    assert r.cart is not None and len(r.cart.lines) == 3


def test_plural_add_only_covers_the_current_page_not_older_pages():
    sm, store = _session([_A, _B])
    store.set_last_shown_products("s1", [_D, _E])  # current page; [_A, _B] is older

    _send(sm, store, "add them", Action(intent=Intent.NO_OP, cart_action="add", generic_reference="plural"))

    assert sorted(_cart_ids(store)) == sorted([_D.id, _E.id])


def test_plural_add_with_an_unpriced_product_adds_nothing():
    unpriced = DisplayedProduct(id=200, name_en="Zulu Mystery Chips", name_ar="زولو", brand_normalized=None,
                                price=None, special_price=None)
    sm, store = _session([_A, unpriced])

    _send(sm, store, "add them", Action(intent=Intent.NO_OP, cart_action="add", generic_reference="plural"))

    assert _cart_ids(store) == []  # all-or-nothing, same purchase-authority rule as any add


def test_plural_add_with_nothing_shown_adds_nothing():
    sm, store = StateManager(), SessionActivityStore()

    r = _send(sm, store, "add them", Action(intent=Intent.NO_OP, cart_action="add", generic_reference="plural"))

    assert _cart_ids(store) == []
    assert r.assistant_message


# --- C: real product evidence still wins / still fails honestly ---

def test_explicit_unmatched_product_name_keeps_not_found_behavior():
    sm, store = _session([_A, _B])
    action = Action(intent=Intent.NO_OP, cart_action="add", selection_text="xyzzyplugh")

    r = _send(sm, store, "add the xyzzyplugh to my cart", action)

    assert "wasn't one of the products" in r.assistant_message.lower()
    assert _cart_ids(store) == []


def test_ordinal_add_is_unaffected():
    sm, store = _session([_A, _B, _C])
    action = Action(intent=Intent.NO_OP, cart_action="add", selection_ordinals=[2])

    _send(sm, store, "add the second one to cart", action)

    assert _cart_ids(store) == [_B.id]


def test_a_named_product_in_the_raw_message_beats_a_generic_flag():
    """generic_reference cannot override real product evidence in the user's
    own words: the raw exact title is still resolved by the backend."""
    sm, store = _session([_A, _B])
    action = Action(intent=Intent.NO_OP, cart_action="add", generic_reference="singular")

    _send(sm, store, "add Bravo Salted Chips - 45G please", action)

    assert _cart_ids(store) == [_B.id]


# --- D: already pending -> the SAME frozen candidates keep governing ---

def test_repeated_generic_add_preserves_the_frozen_candidates():
    sm, store = _session([_A, _B])
    _send(sm, store, "add to cart", Action(intent=Intent.NO_OP, cart_action="add", generic_reference="singular"))
    frozen = list(store.get("s1").pending_action.shown_product_ids)
    assert set(frozen) == {_A.id, _B.id}

    # The live page changes underneath (e.g. results refreshed elsewhere);
    # the open question is still about the ORIGINAL candidates.
    store.set_last_shown_products("s1", [_D, _E])
    _send(sm, store, "just add it", Action(intent=Intent.NO_OP, cart_action="add", generic_reference="singular"))

    pending = store.get("s1").pending_action
    assert pending is not None and pending.kind == "cart_add_pending"
    assert pending.shown_product_ids == frozen
    assert _cart_ids(store) == []


def test_ordinal_answer_resolves_against_the_frozen_candidates_not_the_live_page():
    sm, store = _session([_A, _B])
    _send(sm, store, "add to cart", Action(intent=Intent.NO_OP, cart_action="add", generic_reference="singular"))
    store.set_last_shown_products("s1", [_D, _E])

    # Deterministic pending resolver (0 LLM): understand_fn must not be called.
    handle_message("s1", "the second one", state_manager=sm, activity_store=store,
                   understand_fn=stub_understand(), search_fn=no_search)

    assert _cart_ids(store) == [_B.id]  # frozen [_A, _B] -> #2, NOT the live page's #2 (_E)


def test_plural_reference_while_pending_adds_the_frozen_current_group():
    sm, store = _session([_A, _B])
    _send(sm, store, "add to cart", Action(intent=Intent.NO_OP, cart_action="add", generic_reference="singular"))

    _send(sm, store, "add them all", Action(intent=Intent.NO_OP, cart_action="add", generic_reference="plural"))

    assert _cart_ids(store) == [_A.id, _B.id]
    assert store.get("s1").pending_action is None


def test_plural_reference_while_pending_never_reaches_outside_the_frozen_candidates():
    sm, store = _session([_A, _B])
    _send(sm, store, "add to cart", Action(intent=Intent.NO_OP, cart_action="add", generic_reference="singular"))
    frozen = list(store.get("s1").pending_action.shown_product_ids)
    store.set_last_shown_products("s1", [_D, _E])  # live page diverged from the frozen question

    _send(sm, store, "add them", Action(intent=Intent.NO_OP, cart_action="add", generic_reference="plural"))

    assert _cart_ids(store) == []  # nothing from the unrelated live page is added
    assert store.get("s1").pending_action.shown_product_ids == frozen


def test_pending_generic_continuation_without_cart_verb_is_forced_to_add():
    """The LLM may return only intent=no_op + a generic plural reference for a
    bare "them" while a cart-add question is open; the backend still forces
    the ADD (existing pending-continuation rule, now generic-aware)."""
    sm, store = _session([_A, _B])
    _send(sm, store, "add to cart", Action(intent=Intent.NO_OP, cart_action="add", generic_reference="singular"))

    _send(sm, store, "both", Action(intent=Intent.NO_OP, generic_reference="plural"))

    assert _cart_ids(store) == [_A.id, _B.id]


# --- E: the structured evidence test itself ---

def test_names_a_product_reads_structure_not_wording():
    assert not _names_a_product(Action(intent=Intent.NO_OP, cart_action="add"))
    assert not _names_a_product(Action(intent=Intent.NO_OP, cart_action="add", selection_text="add to cart",
                                       generic_reference="singular"))
    assert _names_a_product(Action(intent=Intent.NO_OP, cart_action="add", selection_text="Pepsi"))
    assert _names_a_product(Action(intent=Intent.NO_OP, cart_action="add", selection_ordinals=[1]))
    assert _names_a_product(Action(intent=Intent.NO_OP, cart_action="add", selection_last=True))


def test_action_schema_only_accepts_singular_or_plural():
    assert Action(intent=Intent.NO_OP, generic_reference="plural").generic_reference == "plural"
    assert Action(intent=Intent.NO_OP).generic_reference is None
    with pytest.raises(Exception):
        Action(intent=Intent.NO_OP, generic_reference="everything")


# --- F: safety net -- the LLM echoes the WHOLE utterance into selection_text
# but does NOT set generic_reference. Structural (whole-string similarity), so
# the wording is irrelevant; real deterministic evidence still wins. ---

@pytest.mark.parametrize("wording", ["add to cart", "حطهم في الكارت", "i want to add it to my cart please!"])
def test_whole_utterance_echoed_as_selection_text_is_treated_as_generic(wording):
    sm, store = _session([_A, _B, _C])
    action = Action(intent=Intent.NO_OP, cart_action="add", selection_text=wording)  # no generic_reference

    r = _send(sm, store, wording, action)

    assert "wasn't one of the products" not in r.assistant_message.lower()
    assert "ده مش من المنتجات" not in r.assistant_message
    assert _cart_ids(store) == []
    pending = store.get("s1").pending_action
    assert pending is not None and pending.kind == "cart_add_pending"
    assert set(pending.shown_product_ids) == {_A.id, _B.id, _C.id}

    # ...and with exactly one shown product the same echo simply adds it.
    sm1, store1 = _session([_A])
    _send(sm1, store1, wording, Action(intent=Intent.NO_OP, cart_action="add", selection_text=wording))
    assert _cart_ids(store1) == [_A.id]


def test_echoed_selection_text_never_overrides_real_product_or_ordinal_evidence():
    # Exact shown title in the raw message (no cart word, so the zero-LLM fast
    # path defers to the LLM path this test exercises).
    msg = "add Bravo Salted Chips - 45G"
    sm, store = _session([_A, _B, _C])
    _send(sm, store, msg, Action(intent=Intent.NO_OP, cart_action="add", selection_text=msg))
    assert _cart_ids(store) == [_B.id]

    # A stated position in the raw message wins too.
    msg = "add the third one to cart"
    sm, store = _session([_A, _B, _C])
    _send(sm, store, msg, Action(intent=Intent.NO_OP, cart_action="add", selection_text=msg, selection_ordinals=[3]))
    assert _cart_ids(store) == [_C.id]


def test_echoed_selection_text_while_pending_keeps_the_frozen_candidates():
    sm, store = _session([_A, _B])
    _send(sm, store, "add to cart", Action(intent=Intent.NO_OP, cart_action="add", generic_reference="singular"))
    frozen = list(store.get("s1").pending_action.shown_product_ids)
    store.set_last_shown_products("s1", [_D, _E])

    _send(sm, store, "put it in the cart", Action(intent=Intent.NO_OP, cart_action="add",
                                                   selection_text="put it in the cart"))

    assert _cart_ids(store) == []
    assert store.get("s1").pending_action.shown_product_ids == frozen
