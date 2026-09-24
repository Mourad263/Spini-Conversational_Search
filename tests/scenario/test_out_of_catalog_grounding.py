"""Sprint 6 out-of-catalog grounding task: the deterministic pre-search
gate (_lacks_catalog_evidence) in src/scenario/orchestrator.py. Hermetic --
understand() always stubbed with a fixed Action (the real regression is
verified separately, live, in the task report); RecordingSearch wraps the
REAL SearchAdapter/DB so tests can assert it was never even called for a
rejected request.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.scenario.orchestrator import _lacks_catalog_evidence, handle_message  # noqa: E402
from src.scenario.session_activity import SessionActivityStore  # noqa: E402
from src.search_adapter.adapter import FilterSet, SearchAdapter, SearchResult  # noqa: E402
from src.state_manager.state_manager import StateManager  # noqa: E402
from src.understanding.schema import Action, Intent  # noqa: E402

LOGGER_NAME = "src.scenario.orchestrator"


def stub_understand(*actions: Action):
    it = iter(actions)

    def _stub(message: str, history: list[str] | None = None) -> Action:
        return next(it)

    return _stub


def _last_trace(caplog) -> dict:
    records = [r for r in caplog.records if r.name == LOGGER_NAME and r.msg == "request_trace %s"]
    assert records, "expected a request_trace log record"
    return records[-1].args


class RecordingSearch:
    def __init__(self) -> None:
        self.calls: list[FilterSet] = []

    def __call__(self, filters: FilterSet, limit: int, offset: int) -> SearchResult:
        self.calls.append(filters)
        return SearchAdapter().search(filters, limit=limit, offset=offset)


# --- unit-level: the gate function itself, no DB/network needed ---

def test_gate_rejects_unsupported_concept_with_one_incidental_shared_word():
    # the real regression, in resolved-filter shape (validate() already ran)
    assert _lacks_catalog_evidence(FilterSet(query_text="كوتش عربية")) is True


def test_gate_passes_when_category_resolved_regardless_of_query_text():
    assert _lacks_catalog_evidence(FilterSet(category="milk_29", query_text="anything")) is False


def test_gate_passes_when_brand_resolved():
    assert _lacks_catalog_evidence(FilterSet(brand="juhayna")) is False


def test_gate_passes_with_no_query_text():
    assert _lacks_catalog_evidence(FilterSet(price_max=50.0)) is False


def test_gate_passes_real_multiword_query_all_tokens_attested():
    assert _lacks_catalog_evidence(FilterSet(query_text="oat milk")) is False


def test_gate_rejects_words_independently_present_but_never_co_occurring():
    # Sprint 6 fuzzy-category-grounding follow-up: "laptop" and "charger"
    # each independently occur in a few unrelated real products, but ZERO
    # real products contain both words together -- independent presence
    # alone (the OLD rule) wrongly let this through; genuine co-occurrence
    # (the NEW rule) correctly rejects it.
    assert _lacks_catalog_evidence(FilterSet(query_text="laptop charger")) is True


# --- A: the known regression, through the full orchestrator path ---

def test_known_regression_car_tires_stops_before_search():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="كوتش عربية"),
    )
    search = RecordingSearch()

    r = handle_message("s1", "انا عاوز كوتش عربيه", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert r.products == []
    assert r.recommendations == []
    assert r.likely_out_of_catalog is True
    assert r.zero_result is True
    assert len(search.calls) == 0  # SearchAdapter never reached
    assert sm.get_state("s1") == FilterSet()  # canonical state untouched


# --- B: unseen unsupported concepts (English + one more), same mechanism ---

def test_unseen_english_unsupported_concept_stops():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(Action(intent=Intent.SEARCH, raw_query_text="bicycle helmet"))
    search = RecordingSearch()

    r = handle_message("s1", "I need a bicycle helmet", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert r.products == []
    assert r.likely_out_of_catalog is True
    assert len(search.calls) == 0


def test_unseen_second_unsupported_concept_stops():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(Action(intent=Intent.SEARCH, raw_query_text="smartphone charger"))
    search = RecordingSearch()

    r = handle_message("s1", "I need a smartphone charger", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert r.products == []
    assert r.likely_out_of_catalog is True
    assert len(search.calls) == 0


# --- C: a real, uncommon catalog product is still allowed through ---

def test_uncommon_real_catalog_product_still_searches():
    # category_hint deliberately left unresolved (None) so this exercises
    # the harder free-text/ALL-tokens-attested branch, not the category
    # shortcut -- "halloumi" is real, corpus-attested vocabulary (>=3 real
    # products; ranking.py's own calibration docstring confirms this).
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(Action(intent=Intent.SEARCH, raw_query_text="halloumi"))
    search = RecordingSearch()

    r = handle_message("s1", "I need halloumi", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert len(search.calls) == 1
    assert r.likely_out_of_catalog is False


# --- D: typo recovery still works (category resolves despite the typo) ---

def test_typo_recovery_via_resolved_category_is_not_blocked():
    sm, store = StateManager(), SessionActivityStore()
    # mirrors tests/understanding/test_llm.py's real, already-passing contract:
    # understand("milq") normalizes raw_query_text to "milk" and category_hint
    # resolves too -- the gate must never re-introduce a typo regression here.
    understand_fn = stub_understand(Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"))
    search = RecordingSearch()

    r = handle_message("s1", "milq", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert len(search.calls) == 1
    assert r.likely_out_of_catalog is False
    assert r.resolved_category == "milk_29"


# --- E/F: Arabic and code-switch real searches still work ---

def test_arabic_real_search_still_works():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(Action(intent=Intent.SEARCH, raw_query_text="جبنة", category_hint="cheese"))
    search = RecordingSearch()

    r = handle_message("s1", "عايز جبنة", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert len(search.calls) == 1
    assert r.likely_out_of_catalog is False


def test_code_switch_real_search_still_works():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(Action(intent=Intent.SEARCH, raw_query_text="oat milk"))
    search = RecordingSearch()

    r = handle_message("s1", "عايز oat milk من فضلك", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert len(search.calls) == 1
    assert r.likely_out_of_catalog is False


# --- G: recommendation regression -- unaffected for a valid, grounded search ---

def test_recommendation_behavior_unaffected_for_valid_search():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk", recommendation_request=True),
    )
    search = RecordingSearch()

    r = handle_message("s1", "what milk do you recommend?", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert len(search.calls) == 1
    assert r.likely_out_of_catalog is False
    # real recommendation service ran (real DB/clustering) -- non-empty for a
    # real, populous category like milk
    assert isinstance(r.recommendations, list)


# --- H: natural container/partitive phrasing ("a bag of chips") must not be
# mistaken for an out-of-catalog concept -- the LLM is expected to extract
# the product itself (raw_query_text="chips"), not the container phrase, so
# these stub Actions reflect that FIXED understanding-layer output. The
# live regression is verified separately in the task report. ---

def test_bag_of_chips_container_phrase_not_out_of_catalog(caplog):
    # Full end-to-end scenario regression (Sprint 6 product-identity
    # cleanup task): given the understanding layer's own fixed, already-
    # verified output for this exact message (raw_query_text="chips", the
    # container word stripped -- see this module's docstring for where
    # the live LLM behavior itself was verified), the turn must reach the
    # normal shopping_search route with a real search call, never the
    # out-of-catalog gate.
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(Action(intent=Intent.SEARCH, raw_query_text="chips", category_hint="chips"))
    search = RecordingSearch()

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        r = handle_message("s1", "i want a bag of chips", state_manager=sm, activity_store=store,
                            understand_fn=understand_fn, search_fn=search)

    assert len(search.calls) == 1
    assert r.likely_out_of_catalog is False
    trace = _last_trace(caplog)
    assert trace["route"] == "shopping_search"
    assert trace["search_calls"] == 1


def test_bag_of_chips_with_price_full_path():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="chips", category_hint="chips", price_max=50.0),
    )
    search = RecordingSearch()

    r = handle_message("s1", "i want a bag of chips under 50", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert len(search.calls) == 1
    assert r.likely_out_of_catalog is False
    assert search.calls[0].price_max == 50.0


def test_potato_chips_multiword_concept_preserved():
    # regression guard: fixing the container-phrase case must not turn a
    # genuine multi-word product concept into just its final noun.
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(Action(intent=Intent.SEARCH, raw_query_text="potato chips", price_max=50.0))
    search = RecordingSearch()

    r = handle_message("s1", "potato chips under 50", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert len(search.calls) == 1
    assert r.likely_out_of_catalog is False


def test_multivitamin_tablets_correctly_rejected_no_catalog_evidence():
    # Sprint 6 multivitamin investigation: authoritative catalog check
    # (real DB scan) found ZERO products under "multivitamin" in any
    # spelling, ZERO dietary-supplement tablets (the only "tablet(s)"
    # matches are dishwasher/laundry/sweetener/charcoal tablets), and no
    # pharmacy/supplement/vitamin canonical category at all -- every real
    # "vitamin" match is a cosmetic/food product listing it as an
    # ingredient (shampoo, face cream, hair oil, cocoa powder). This
    # catalog genuinely does not sell vitamin supplements, under any
    # wording -- out-of-catalog is the CORRECT result, not a grounding
    # bug. raw_query_text reflects the understanding layer's own
    # container-word-stripping/typo-normalization rules (see this
    # module's docstring for how that's verified) -- "tablets" is real
    # product-FORM identity here (kept, like "potato chips" keeps
    # "potato"), not a stripped container word.
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(Action(intent=Intent.SEARCH, raw_query_text="multivitamin tablets"))
    search = RecordingSearch()

    r = handle_message("s1", "i want a multivitamen tablets", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert len(search.calls) == 0
    assert r.likely_out_of_catalog is True


def test_laptop_charger_full_path_still_rejected():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(Action(intent=Intent.SEARCH, raw_query_text="laptop charger"))
    search = RecordingSearch()

    r = handle_message("s1", "laptop charger", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert len(search.calls) == 0
    assert r.likely_out_of_catalog is True


def test_car_tires_full_path_still_rejected():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(Action(intent=Intent.SEARCH, raw_query_text="car tires"))
    search = RecordingSearch()

    r = handle_message("s1", "car tires", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert len(search.calls) == 0
    assert r.likely_out_of_catalog is True


# --- state safety: a rejected ADD_FILTER never overwrites the active search ---

def test_rejected_add_filter_does_not_touch_existing_active_state():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk", price_max=50.0),
        Action(intent=Intent.ADD_FILTER, raw_query_text="كوتش عربية"),
    )
    search = RecordingSearch()
    handle_message("s1", "milk under 50", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    state_before = sm.get_state("s1")

    r = handle_message("s1", "and also كوتش عربية", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert r.likely_out_of_catalog is True
    assert sm.get_state("s1") == state_before  # milk/50 filter untouched
