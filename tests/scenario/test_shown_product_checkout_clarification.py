"""Sprint 6 shown-product checkout-clarification task: real browser bug --
several Pepsi variants shown, "i want to buy it" correctly asked for
clarification, but then the user supplying the EXACT shown title
("Pepsi - Carbonated Drink - 1.5L") STILL returned "more than one match,"
and the already-established buy/checkout intent was lost (the user had
to say "buy" again). A copied bilingual product card also flipped the
conversation to Arabic.

Three general, in-place fixes, all reusing existing mechanisms:

1. src/scenario/orchestrator.py's _resolve_product_selection(): an EXACT
   normalized match against a shown product's own name_en/name_ar now
   wins outright, before fuzzy WRatio scoring ever runs. Proven root
   cause: the exact shown title scores 100 against itself but ALSO >= 80
   against several other real shown variants (fuzzy WRatio rewards
   shared "Pepsi Carbonated Drink..." wording), so a perfect match was
   being reported ambiguous under the threshold-only fuzzy pass.

2. Pending checkout intent now reuses the EXISTING single-slot
   PendingAction/pending_action mechanism (SessionActivityStore's
   set_pending_recommendation_offer generalized to set_pending_action(
   kind, ...), popped unconditionally by the SAME A2.5 code the
   recommendation-offer task already added) -- no new session field.
   When a checkout-requested turn's selection is still ambiguous,
   kind="checkout_pending_selection" is stored; the next NO_OP turn (never
   a genuine new search/filter turn) treats checkout as still requested
   even if the LLM doesn't re-state checkout_request itself.

3. src/ingestion/text_normalize.py's is_arabic() now requires Arabic to
   be the DOMINANT script (more Arabic characters than Latin letters),
   not merely present anywhere -- a single-script input (any real
   message or catalog field) is unaffected either way; only a genuinely
   mixed message (English conversation + a pasted catalog Arabic name)
   changes, and correctly stays English.

Hermetic -- `understand_fn` is always stubbed; a controlled, realistic
DisplayedProduct fixture set is used (real Pepsi/Dove/Fa-shaped rows),
matching the established pattern from the product-selection/checkout
tasks. RecordingSearch wraps the real SearchAdapter/DB only where a
genuine search must run.
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


_PEPSI = [
    DisplayedProduct(id=1, name_en="Pepsi - Carbonated Drink - 1.5L",
                      name_ar="بيبسي مشروب غازى، 1.5 لتر", brand_normalized="pepsi", price=24.95, special_price=24.95),
    DisplayedProduct(id=2, name_en="Pepsi Carbonated Drink - 400Ml",
                      name_ar="بيبسي مشروب غازى، 400 مل", brand_normalized="pepsi", price=12.5, special_price=12.5),
    DisplayedProduct(id=3, name_en="Pepsi Can- 330Ml",
                      name_ar="بيبسي كانز، 330 مل", brand_normalized="pepsi", price=9.95, special_price=9.95),
    DisplayedProduct(id=4, name_en="Pepsi Diet Carbonated Drink- 400Ml",
                      name_ar="بيبسي دايت مشروب غازى، 400 مل", brand_normalized="pepsi", price=12.5, special_price=12.5),
    DisplayedProduct(id=5, name_en="Pepsi Cola-Diet Can 330Ml",
                      name_ar="بيبسي كولا دايت كانز 330 مل", brand_normalized="pepsi", price=9.95, special_price=9.95),
]

_DOVE = [
    DisplayedProduct(id=201, name_en="Dove Deodorant Stick", name_ar="دوف مزيل عرق استيك", brand_normalized="dove"),
    DisplayedProduct(id=202, name_en="Dove Deodorant Spray", name_ar="دوف مزيل عرق سبراي", brand_normalized="dove"),
]

_FA = [
    DisplayedProduct(id=301, name_en="Fa Yoghurt Shower Cream With Vanilla Honey",
                      name_ar="فا كريم استحمام بالزبادي والعسل والفانيليا", brand_normalized="fa"),
    DisplayedProduct(id=302, name_en="Fa Fresh Orange Shower Gel", name_ar="فا جل استحمام البرتقال المنعش", brand_normalized="fa"),
]


def _primed_session(session_id: str, shown: list[DisplayedProduct]) -> tuple[StateManager, SessionActivityStore]:
    sm, store = StateManager(), SessionActivityStore()
    # Cart/direct-checkout architecture task: use the canonical setter so
    # recent_shown_pages (consulted by named/raw-message resolution) is
    # populated too, not just last_shown_products.
    store.set_last_shown_products(session_id, shown)
    return sm, store


# --- A: ambiguous "buy it" -> clarification requested, checkout intent preserved ---

def test_ambiguous_buy_it_asks_for_clarification():
    sm, store = _primed_session("s1", _PEPSI)
    understand_fn = stub_understand(Action(intent=Intent.NO_OP, selection_text="Pepsi", checkout_request=True))
    search = RecordingSearch()

    r = handle_message("s1", "i want to buy it", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert "more than one match" in r.assistant_message.lower()
    assert store.get("s1").checkout_active is False
    assert store.get("s1").pending_action is not None
    assert store.get("s1").pending_action.kind == "checkout_pending_selection"


# --- B: exact full shown English title -> unique selection, checkout opens
# immediately, even when the LLM does NOT restate checkout_request ---

def test_exact_full_title_resolves_and_opens_checkout_without_restated_intent():
    sm, store = _primed_session("s1", _PEPSI)
    understand_fn = stub_understand(
        Action(intent=Intent.NO_OP, selection_text="Pepsi", checkout_request=True),
        Action(intent=Intent.NO_OP, selection_text="Pepsi - Carbonated Drink - 1.5L"),  # no checkout_request here
    )
    search = RecordingSearch()

    handle_message("s1", "i want to buy it", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    r2 = handle_message("s1", "Pepsi - Carbonated Drink - 1.5L", state_manager=sm, activity_store=store,
                         understand_fn=understand_fn, search_fn=search)

    assert store.get("s1").selected_products[0].product.id == 1
    assert r2.checkout_form is not None
    assert r2.checkout_form.lines[0].id == 1
    assert store.get("s1").checkout_active is True
    assert store.get("s1").pending_action is None


# --- C/D: ordinal / "the third one" clarification after ambiguous buy also
# opens checkout ---

def test_ordinal_clarification_after_ambiguous_buy_opens_checkout():
    sm, store = _primed_session("s1", _PEPSI)
    understand_fn = stub_understand(
        Action(intent=Intent.NO_OP, selection_text="Pepsi", checkout_request=True),
        Action(intent=Intent.NO_OP, selection_ordinals=[3]),
    )
    search = RecordingSearch()

    handle_message("s1", "i want to buy it", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    r2 = handle_message("s1", "3", state_manager=sm, activity_store=store,
                         understand_fn=understand_fn, search_fn=search)

    assert store.get("s1").selected_products[0].product.id == 3
    assert r2.checkout_form is not None
    assert r2.checkout_form.lines[0].id == 3


def test_the_third_one_clarification_after_ambiguous_buy_opens_checkout():
    sm, store = _primed_session("s1", _PEPSI)
    understand_fn = stub_understand(
        Action(intent=Intent.NO_OP, selection_text="Pepsi", checkout_request=True),
        Action(intent=Intent.NO_OP, selection_ordinals=[3]),
    )
    search = RecordingSearch()

    handle_message("s1", "i want to buy it", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    r2 = handle_message("s1", "the third one", state_manager=sm, activity_store=store,
                         understand_fn=understand_fn, search_fn=search)

    assert r2.checkout_form is not None
    assert r2.checkout_form.lines[0].id == 3


# --- E: full copied bilingual product card resolves the correct shown product ---

def test_bilingual_card_with_distinguishing_text_resolves_uniquely():
    sm, store = _primed_session("s1", _PEPSI)
    understand_fn = stub_understand(
        Action(intent=Intent.NO_OP, selection_text="Pepsi - Carbonated Drink - 1.5L", checkout_request=True),
    )
    search = RecordingSearch()

    r = handle_message(
        "s1", "Pepsi - Carbonated Drink - 1.5L\nبيبسي مشروب غازى، 1.5 لتر\nPepsi\n24.95 EGP",
        state_manager=sm, activity_store=store, understand_fn=understand_fn, search_fn=search,
    )

    assert store.get("s1").selected_products[0].product.id == 1
    assert r.checkout_form is not None


# --- F: English conversation + bilingual product copy -> response stays English ---

def test_bilingual_card_response_stays_english():
    sm, store = _primed_session("s1", _PEPSI)
    understand_fn = stub_understand(
        Action(intent=Intent.NO_OP, selection_text="Pepsi - Carbonated Drink - 1.5L", checkout_request=True),
    )
    search = RecordingSearch()

    r = handle_message(
        "s1", "Pepsi - Carbonated Drink - 1.5L\nبيبسي مشروب غازى، 1.5 لتر\nPepsi\n24.95 EGP",
        state_manager=sm, activity_store=store, understand_fn=understand_fn, search_fn=search,
    )

    assert "checking out" in r.assistant_message.lower()  # English wording, not Arabic


# --- G: a genuine Arabic clarification still works (selection + checkout + Arabic reply) ---

def test_genuine_arabic_clarification_still_works():
    sm, store = _primed_session("s1", _PEPSI)
    understand_fn = stub_understand(
        Action(intent=Intent.NO_OP, selection_text="Pepsi", checkout_request=True),
        Action(intent=Intent.NO_OP, selection_ordinals=[3], checkout_request=True),
    )
    search = RecordingSearch()

    handle_message("s1", "i want to buy it", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    r2 = handle_message("s1", "عايز التالت", state_manager=sm, activity_store=store,
                         understand_fn=understand_fn, search_fn=search)

    assert store.get("s1").selected_products[0].product.id == 3
    assert r2.checkout_form is not None
    assert "تمام" in r2.assistant_message  # Arabic checkout wording


# --- H: partial same-brand reference remains ambiguous (ambiguity safety) ---

def test_partial_same_brand_reference_remains_ambiguous():
    sm, store = _primed_session("s1", _DOVE)
    understand_fn = stub_understand(Action(intent=Intent.NO_OP, selection_text="Dove"))
    search = RecordingSearch()

    r = handle_message("s1", "Dove", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert store.get("s1").selected_products == []
    assert "more than one match" in r.assistant_message.lower()


# --- I: unmatched full name -> no global catalog fallback search ---

def test_unmatched_reference_never_triggers_a_catalog_search():
    sm, store = _primed_session("s1", _PEPSI)
    understand_fn = stub_understand(Action(intent=Intent.NO_OP, selection_text="xyzzyplughnonexistent", checkout_request=True))
    search = RecordingSearch()

    r = handle_message("s1", "xyzzyplughnonexistent", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert len(search.calls) == 0
    assert store.get("s1").selected_products == []
    assert store.get("s1").checkout_active is False


# --- J: a new successful search supersedes stale pending checkout intent ---

def test_new_search_clears_stale_pending_checkout():
    sm, store = _primed_session("s1", _PEPSI)
    understand_fn = stub_understand(
        Action(intent=Intent.NO_OP, selection_text="Pepsi", checkout_request=True),
        Action(intent=Intent.SEARCH, raw_query_text="chocolate", category_hint="chocolate"),
        Action(intent=Intent.NO_OP, selection_ordinals=[1]),
    )
    search = RecordingSearch()

    handle_message("s1", "i want to buy it", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    assert store.get("s1").pending_action is not None

    handle_message("s1", "show me chocolate", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    shown = store.get("s1").last_shown_products
    assert shown[0].id not in {p.id for p in _PEPSI}  # a real, different product set

    # Cart/direct-checkout architecture task: this 3rd turn is a PLAIN
    # (non-checkout) selection -- it does not write selected_products,
    # which is reserved for an actually-opened checkout draft.
    r3 = handle_message("s1", "the first one", state_manager=sm, activity_store=store,
                         understand_fn=understand_fn, search_fn=search)

    assert store.get("s1").selected_products == []
    assert shown[0].name_en in r3.assistant_message
    assert r3.checkout_form is None  # NOT the old stale Pepsi checkout


# --- K: RESET / session expiry clears pending checkout ---

def test_reset_clears_pending_checkout():
    sm, store = _primed_session("s1", _PEPSI)
    understand_fn = stub_understand(
        Action(intent=Intent.NO_OP, selection_text="Pepsi", checkout_request=True),
        Action(intent=Intent.RESET),
    )
    search = RecordingSearch()

    handle_message("s1", "i want to buy it", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    assert store.get("s1").pending_action is not None

    handle_message("s1", "start over", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)

    assert store.get("s1").pending_action is None
    assert store.get("s1").checkout_active is False


def test_session_expiry_clears_pending_checkout():
    sm, store = _primed_session("s1", _PEPSI)
    understand_fn = stub_understand(
        Action(intent=Intent.NO_OP, selection_text="Pepsi", checkout_request=True),
        Action(intent=Intent.OFF_TOPIC),
    )
    search = RecordingSearch()

    handle_message("s1", "i want to buy it", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search, now=0.0)
    assert store.get("s1", now=0.0).pending_action is not None

    from src.scenario.session_activity import SESSION_TTL_SECONDS
    handle_message("s1", "hello", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search, now=SESSION_TTL_SECONDS + 100.0)

    assert store.get("s1", now=SESSION_TTL_SECONDS + 100.0).pending_action is None


# --- L: same-turn "select X and checkout" regression ---

def test_same_turn_selection_and_checkout_regression():
    sm, store = _primed_session("s1", _PEPSI)
    understand_fn = stub_understand(
        Action(intent=Intent.NO_OP, selection_ordinals=[2], checkout_request=True),
    )
    search = RecordingSearch()

    r = handle_message("s1", "I'll take the second one and checkout", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert store.get("s1").selected_products[0].product.id == 2
    assert r.checkout_form is not None
    assert r.checkout_form.lines[0].id == 2


# --- M: "fa yoghart" fuzzy named-selection regression -- same shown-set
# shape as the original task (one Fa product among unrelated products;
# see test_named_product_selection.py, unaffected by this task's fixes) ---

def test_fa_yoghart_fuzzy_selection_regression():
    sm, store = _primed_session("s1", [_FA[0], _DOVE[0]])
    understand_fn = stub_understand(Action(intent=Intent.NO_OP, selection_text="fa yoghart"))
    search = RecordingSearch()

    r = handle_message("s1", "i will buy the fa yoghart", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    # Cart/direct-checkout architecture task: a PLAIN (non-checkout)
    # selection does not write selected_products.
    assert store.get("s1").selected_products == []
    assert _FA[0].name_en in r.assistant_message


# --- N: ordinal selection regression (no checkout involved) ---

def test_ordinal_selection_regression_no_checkout():
    sm, store = _primed_session("s1", _PEPSI)
    understand_fn = stub_understand(Action(intent=Intent.NO_OP, selection_ordinals=[2]))
    search = RecordingSearch()

    r = handle_message("s1", "the second one", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    # Cart/direct-checkout architecture task: a PLAIN (non-checkout)
    # selection does not write selected_products.
    assert store.get("s1").selected_products == []
    assert _PEPSI[1].name_en in r.assistant_message
    assert r.checkout_form is None


# --- generalization: two additional real-shaped displayed-product sets ---

def test_generalization_dove_exact_title_unique_selection():
    sm, store = _primed_session("s1", _DOVE)
    understand_fn = stub_understand(Action(intent=Intent.NO_OP, selection_text="Dove Deodorant Stick"))
    search = RecordingSearch()

    r = handle_message("s1", "Dove Deodorant Stick", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    # Cart/direct-checkout architecture task: a PLAIN (non-checkout)
    # selection does not write selected_products.
    assert store.get("s1").selected_products == []
    assert _DOVE[0].name_en in r.assistant_message
    assert "more than one match" not in r.assistant_message.lower()


def test_generalization_fa_exact_title_unique_selection():
    sm, store = _primed_session("s1", _FA)
    understand_fn = stub_understand(
        Action(intent=Intent.NO_OP, selection_text="Fa Yoghurt Shower Cream With Vanilla Honey"),
    )
    search = RecordingSearch()

    r = handle_message("s1", "Fa Yoghurt Shower Cream With Vanilla Honey", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    # Cart/direct-checkout architecture task: a PLAIN (non-checkout)
    # selection does not write selected_products.
    assert store.get("s1").selected_products == []
    assert _FA[0].name_en in r.assistant_message
    assert "more than one match" not in r.assistant_message.lower()


def test_generalization_fa_brand_only_reference_remains_ambiguous():
    sm, store = _primed_session("s1", _FA)
    understand_fn = stub_understand(Action(intent=Intent.NO_OP, selection_text="Fa"))
    search = RecordingSearch()

    r = handle_message("s1", "Fa", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert store.get("s1").selected_products == []
    assert "more than one match" in r.assistant_message.lower()


# ======================================================================
# Raw-message shown-product authority (real browser bug): the LLM can
# correctly flag a turn as a selection while still mis-copying WHICH
# product into Action.selection_text. A complete shown title clearly
# present in the user's own RAW message must win over that paraphrase.
# ======================================================================

_SPICY = DisplayedProduct(
    id=401, name_en="Bake Away Frozen Spicy Tortilla 8 Inc 6 Pcs - 220G",
    name_ar="بيك اواى تورتيلا سبايسى 8 انش 6 قطع - 220جرام", brand_normalized="bake away",
    price=59.95, special_price=59.95,
)
_WHEAT = DisplayedProduct(
    id=402, name_en="Bake Away Frozen Whole Wheat Tortilla 8 Inc 6 Pcs - 220G",
    name_ar="بيك اواى تورتيلا قمح كامل 8 انش 6 قطع - 220جرام", brand_normalized="bake away",
    price=59.95, special_price=59.95,
)
_TORTILLA = [
    _SPICY, _WHEAT,
    DisplayedProduct(id=403, name_en="Freska Frozen Plain Tortilla 10 Pcs - 400G",
                      name_ar="فريسكا تورتيلا سادة", brand_normalized="freska", price=45.0, special_price=45.0),
    DisplayedProduct(id=404, name_en="Americana Tortilla Wraps 6 Pcs",
                      name_ar="امريكانا تورتيلا رول", brand_normalized="americana", price=30.0, special_price=30.0),
]


# --- H: raw exact Spicy title wins over an incorrect/lossy Whole-Wheat Action ---

def test_raw_exact_title_overrides_wrong_llm_selection_text():
    sm, store = _primed_session("s1", _TORTILLA)
    understand_fn = stub_understand(
        # Simulates a real LLM extraction error: the model reports the
        # OTHER shown product even though the user typed the Spicy title
        # verbatim -- the raw message must still win.
        Action(intent=Intent.NO_OP, selection_text=_WHEAT.name_en, checkout_request=True),
    )
    search = RecordingSearch()

    r = handle_message("s1", _SPICY.name_en, state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert store.get("s1").selected_products[0].product.id == _SPICY.id
    assert r.checkout_form is not None
    assert r.checkout_form.lines[0].id == _SPICY.id
    assert r.checkout_form.lines[0].name_en == _SPICY.name_en


# --- I: bilingual Spicy card resolves uniquely even with a generic Action ---

def test_bilingual_spicy_card_resolves_uniquely_with_generic_action():
    sm, store = _primed_session("s1", _TORTILLA)
    understand_fn = stub_understand(
        Action(intent=Intent.NO_OP, selection_text="Bake Away", checkout_request=True),
    )
    search = RecordingSearch()

    raw = f"{_SPICY.name_en}\n{_SPICY.name_ar}\n59.95 EGP"
    r = handle_message("s1", raw, state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert store.get("s1").selected_products[0].product.id == _SPICY.id
    assert r.checkout_form is not None
    assert r.checkout_form.lines[0].id == _SPICY.id


# --- J: raw message names TWO full shown product titles -> multi-product
# purchase requires the cart (Correction: MULTI-PRODUCT PURCHASE = CART
# ONLY -- a raw message naming 2+ shown products is never a direct
# multi-line checkout; neither product is in the cart yet, so both are
# proposed for cart-add instead of any product being silently checked
# out) ---

def test_two_full_shown_titles_in_raw_message_proposes_cart_add():
    sm, store = _primed_session("s1", _PEPSI)
    understand_fn = stub_understand(Action(intent=Intent.NO_OP, selection_text="Pepsi", checkout_request=True))
    search = RecordingSearch()

    raw = f"i want to buy {_PEPSI[0].name_en} and {_PEPSI[1].name_en}"
    r = handle_message("s1", raw, state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert store.get("s1").selected_products == []
    assert store.get("s1").checkout_active is False
    assert r.checkout_form is None
    assert store.get("s1").pending_action is not None
    assert store.get("s1").pending_action.kind == "multi_add_to_cart_pending"
    assert set(store.get("s1").pending_action.shown_product_ids) == {_PEPSI[0].id, _PEPSI[1].id}
    assert "add" in r.assistant_message.lower() and "cart" in r.assistant_message.lower()


# --- J2: a bare fuzzy/ambiguous reference still never silently proceeds --
# "fail safely using existing selection behavior" for the un-resolvable case ---

def test_bare_ambiguous_reference_never_silently_checks_out_a_subset():
    sm, store = _primed_session("s1", _DOVE)
    understand_fn = stub_understand(Action(intent=Intent.NO_OP, selection_text="Dove", checkout_request=True))
    search = RecordingSearch()

    r = handle_message("s1", "Dove", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert "more than one match" in r.assistant_message.lower()
    assert r.checkout_form is None
    assert store.get("s1").selected_products == []
    assert store.get("s1").checkout_active is False


# --- K: pending checkout + exact clarification still works when the
# clarifying turn's Action mis-copies the product (server-authority chain) ---

def test_pending_checkout_with_wrong_action_still_selects_raw_evidence_product():
    sm, store = _primed_session("s1", _TORTILLA)
    understand_fn = stub_understand(
        Action(intent=Intent.NO_OP, selection_text="Bake Away", checkout_request=True),
        Action(intent=Intent.NO_OP, selection_text=_WHEAT.name_en),  # no checkout_request restated, and wrong product
    )
    search = RecordingSearch()

    handle_message("s1", "i want to buy it", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    r2 = handle_message("s1", _SPICY.name_en, state_manager=sm, activity_store=store,
                         understand_fn=understand_fn, search_fn=search)

    assert store.get("s1").pending_action is None
    assert store.get("s1").selected_products[0].product.id == _SPICY.id
    assert store.get("s1").checkout_active is True
    assert r2.checkout_form is not None
    assert r2.checkout_form.lines[0].id == _SPICY.id


# --- L: selected_products, and checkout_form.lines all point at the SAME
# resolved product -- the persisted order trusts this chain ---

def test_selection_snapshot_and_checkout_form_all_agree_on_spicy():
    sm, store = _primed_session("s1", _TORTILLA)
    understand_fn = stub_understand(
        Action(intent=Intent.NO_OP, selection_text=_WHEAT.name_en, checkout_request=True),
    )
    search = RecordingSearch()

    r = handle_message("s1", _SPICY.name_en, state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    activity = store.get("s1")
    assert activity.selected_products[0].product.id == _SPICY.id
    assert activity.selected_products[0].product.name_en == _SPICY.name_en
    assert r.checkout_form.lines[0].id == _SPICY.id
    assert r.checkout_form.lines[0].name_en == _SPICY.name_en
    # never Whole Wheat anywhere in the chain
    assert activity.selected_products[0].product.id != _WHEAT.id
    assert r.checkout_form.lines[0].id != _WHEAT.id


# ======================================================================
# Vague product-reference safety: a real confirmed live-Groq Action for
# "yes i want to buy it" against several shown Pepsi products came back
# as selection_last=True AND selection_text="Pepsi" together -- the model
# had no real positional evidence for "it" and only a bare, ambiguous
# brand as its own corroborating claim, yet the old resolver trusted
# selection_last outright and silently checked out the wrong (last)
# product. The resolver now only trusts selection_last/selection_ordinal
# when there's no co-present selection_text, or when selection_text
# independently corroborates the SAME product.
# ======================================================================

# --- M: vague "it" (selection_last=True + an ambiguous co-present
# selection_text) with multiple shown products -> ambiguous, pending
# checkout preserved, no product silently selected, zero search calls ---

def test_vague_it_with_uncorroborated_last_stays_ambiguous():
    sm, store = _primed_session("s1", _PEPSI)
    understand_fn = stub_understand(
        Action(intent=Intent.NO_OP, selection_last=True, selection_text="Pepsi", checkout_request=True),
    )
    search = RecordingSearch()

    r = handle_message("s1", "yes i want to buy it", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert "more than one match" in r.assistant_message.lower()
    assert r.checkout_form is None
    assert store.get("s1").selected_products == []
    assert store.get("s1").checkout_active is False
    assert len(search.calls) == 0
    assert store.get("s1").pending_action is not None
    assert store.get("s1").pending_action.kind == "checkout_pending_selection"


# --- N: a genuine "the last one" (selection_last=True, no competing
# selection_text) still selects the REAL last shown product ---

def test_genuine_last_one_reference_still_selects_last_product():
    sm, store = _primed_session("s1", _PEPSI)
    understand_fn = stub_understand(Action(intent=Intent.NO_OP, selection_last=True))
    search = RecordingSearch()

    r = handle_message("s1", "the last one", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    # Cart/direct-checkout architecture task: a PLAIN (non-checkout)
    # selection does not write selected_products.
    assert store.get("s1").selected_products == []
    assert _PEPSI[-1].name_en in r.assistant_message


# --- O: a genuine ordinal reference still selects the REAL third product ---

def test_genuine_ordinal_reference_still_selects_third_product():
    sm, store = _primed_session("s1", _PEPSI)
    understand_fn = stub_understand(Action(intent=Intent.NO_OP, selection_ordinals=[3]))
    search = RecordingSearch()

    r = handle_message("s1", "third one", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    # Cart/direct-checkout architecture task: a PLAIN (non-checkout)
    # selection does not write selected_products.
    assert store.get("s1").selected_products == []
    assert _PEPSI[2].name_en in r.assistant_message


# --- P: ordinal + checkout in one turn still opens checkout directly ---

def test_ordinal_with_checkout_request_opens_checkout_directly():
    sm, store = _primed_session("s1", _PEPSI)
    understand_fn = stub_understand(Action(intent=Intent.NO_OP, selection_ordinals=[3], checkout_request=True))
    search = RecordingSearch()

    r = handle_message("s1", "i want to buy the third one", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert r.checkout_form is not None
    assert r.checkout_form.lines[0].id == _PEPSI[2].id
    assert len(search.calls) == 0


# --- Q: exactly ONE shown product -- the same vague "it" shape may
# directly open checkout, since a single shown product is trivially the
# only possible answer (not LLM trust -- there's no other candidate to
# be wrong about) ---

def test_vague_it_with_single_shown_product_opens_checkout():
    sm, store = _primed_session("s1", [_PEPSI[0]])
    understand_fn = stub_understand(
        Action(intent=Intent.NO_OP, selection_last=True, selection_text="Pepsi", checkout_request=True),
    )
    search = RecordingSearch()

    r = handle_message("s1", "yes i want to buy it", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert r.checkout_form is not None
    assert r.checkout_form.lines[0].id == _PEPSI[0].id
    assert len(search.calls) == 0


# ======================================================================
# Positional raw-message authority: selection_last/selection_ordinal must
# be validated against the RAW USER MESSAGE itself, never cross-checked
# against another LLM field (selection_text) -- validating one LLM field
# with another still leaves product identity entirely LLM-controlled.
# ======================================================================

# --- R: the important missing safety test -- even when selection_text
# happens to be a real, EXACT, uniquely-matching shown product, a vague
# "it" with no raw positional evidence must still be ambiguous, because
# the raw message itself never actually named that product either ---

def test_vague_it_with_exact_but_unraised_selection_text_stays_ambiguous():
    sm, store = _primed_session("s1", _PEPSI)
    understand_fn = stub_understand(
        Action(intent=Intent.NO_OP, selection_last=True, selection_text=_PEPSI[0].name_en, checkout_request=True),
    )
    search = RecordingSearch()

    r = handle_message("s1", "yes i want to buy it", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert "more than one match" in r.assistant_message.lower()
    assert r.checkout_form is None
    assert store.get("s1").selected_products == []
    assert len(search.calls) == 0
    assert store.get("s1").pending_action is not None
    assert store.get("s1").pending_action.kind == "checkout_pending_selection"


# --- S: raw message DOES carry positional evidence ("the last Pepsi") --
# selection_last is trusted and the real last product is selected ---

def test_raw_message_with_last_evidence_selects_last_product():
    sm, store = _primed_session("s1", _PEPSI)
    understand_fn = stub_understand(
        Action(intent=Intent.NO_OP, selection_last=True, selection_text="Pepsi"),
    )
    search = RecordingSearch()

    r = handle_message("s1", "buy the last Pepsi", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    # Cart/direct-checkout architecture task: a PLAIN (non-checkout)
    # selection does not write selected_products.
    assert store.get("s1").selected_products == []
    assert _PEPSI[-1].name_en in r.assistant_message
    assert len(search.calls) == 0
