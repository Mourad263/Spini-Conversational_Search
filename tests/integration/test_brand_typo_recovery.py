"""Sprint 6 brand typo-recovery task: real live browser conversation
exposed that "pebsi" (a one-substitution typo of the real brand "Pepsi")
recovered correctly when typed bare, but the SAME word failed to ground
as a brand inside a natural sentence ("how you cant find pebsi" -- the
live Action for this exact message already sets brand_text="Pepsi"
correctly, confirmed via one live call, so the bug is entirely
downstream of understanding()).

Root cause, proven directly: resolve_brand_text()'s own corpus-protection
safety net (added for a DIFFERENT failure shape -- "mobile phone"
matching the unrelated category "Honey") raises the effective fuzzy
threshold from 80 to 82 whenever the INPUT word itself has zero corpus
attestation -- which is true of essentially any genuine typo. "pebsi"
scores exactly 80 against "pepsi" (a real, 14-product-attested brand) and
was being rejected by this stricter floor, even though
ranking.py's SEPARATE, already-proven-safe spelling corrector
(correct_unmatched_terms, gated on the CANDIDATE's own real-document
count, not the input's) already recovers "pebsi"->"pepsi" correctly for
free-text search. Two independent, disagreeing fuzzy paths for the same
underlying correction question.

General fix: resolve_brand_text() now falls back to
correct_unmatched_terms() when its own direct fuzzy attempt finds
nothing, then re-validates the correction through the SAME corpus-
protected fuzzy match (so it can only ever resolve to a brand that
already independently clears every existing safety net -- confirmed
directly that this does NOT reopen the "phone"->"honey" collision, since
"honey" itself is not a real brand name).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.scenario.orchestrator import handle_message  # noqa: E402
from src.scenario.session_activity import SessionActivityStore  # noqa: E402
from src.search_adapter.adapter import FilterSet, SearchAdapter, SearchResult  # noqa: E402
from src.search_engine.keyword_baseline import resolve_brand_text  # noqa: E402
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


# --- A: exact brand control (unaffected baseline) ---

def test_exact_brand_still_resolves():
    assert resolve_brand_text("Pepsi") == "pepsi"
    assert resolve_brand_text("pepsi") == "pepsi"


# --- B: the proven one-substitution typo now recovers ---

def test_one_substitution_typo_recovers_to_real_brand():
    assert resolve_brand_text("pebsi") == "pepsi"


# --- C: a genuinely weak/ambiguous typo is NOT over-corrected ---

def test_weak_two_substitution_typo_does_not_get_invented():
    assert resolve_brand_text("bebsi") is None


# --- D: the original "mobile phone" -> "Honey" regression stays fixed --
# correct_unmatched_terms("phone") does return "honey" (a real, well-
# attested corpus word), but "honey" is not itself a real brand, so the
# re-validation pass must still correctly return None. ---

def test_phone_honey_collision_not_reintroduced():
    assert resolve_brand_text("phone") is None


# --- E: two additional real-catalog generalization controls (not Pepsi) ---

def test_heinz_one_substitution_typo_recovers():
    assert resolve_brand_text("heins") == "heinz"


def test_dettol_one_substitution_typo_recovers():
    assert resolve_brand_text("dettel") == "dettol"


# --- F: fresh-session bare typo, through the full conversational path ---

def test_fresh_session_bare_typo_resolves_to_real_pepsi_products():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="pebsi", brand_text="pebsi", category_hint="soft drink"),
    )
    search = RecordingSearch()

    r = handle_message("s1", "pebsi", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert r.resolved_brand == "pepsi"
    assert r.products
    assert all(p.brand_normalized == "pepsi" for p in r.products)


# --- G: natural "bottle of" phrasing with the typo recovers the same way ---

def test_bottle_of_typo_phrasing_resolves_to_real_pepsi():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="pebsi", brand_text="pebsi", category_hint="soft drink"),
    )
    search = RecordingSearch()

    r = handle_message("s1", "now i want bottel of pebsi", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert r.resolved_brand == "pepsi"
    assert r.products
    assert all(p.brand_normalized == "pepsi" for p in r.products)


# --- H: conversational correction phrasing, using the REAL confirmed
# Action shape (raw_query_text/brand_text already normalized to "pepsi"/
# "Pepsi" by the understanding layer -- confirmed via one live Groq call) ---

def test_conversational_correction_phrasing_resolves_to_real_pepsi():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="pepsi", brand_text="Pepsi", category_hint="soft drink"),
    )
    search = RecordingSearch()

    r = handle_message("s1", "how you cant find pebsi", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert r.resolved_brand == "pepsi"
    assert r.products
    assert all(p.brand_normalized == "pepsi" for p in r.products)
    assert not r.likely_out_of_catalog


# --- I: a failed weak-typo turn must not contaminate a later corrected one ---

def test_failed_typo_then_corrected_typo_session_isolation():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="bebsi", brand_text="bebsi", category_hint="soft drink"),
        Action(intent=Intent.SEARCH, raw_query_text="pebsi", brand_text="pebsi", category_hint="soft drink"),
    )
    search = RecordingSearch()

    r1 = handle_message("s1", "bebsi", state_manager=sm, activity_store=store,
                         understand_fn=understand_fn, search_fn=search)
    assert r1.resolved_brand is None  # weak typo correctly not resolved

    r2 = handle_message("s1", "pebsi", state_manager=sm, activity_store=store,
                         understand_fn=understand_fn, search_fn=search)
    assert r2.resolved_brand == "pepsi"  # the later, stronger typo still recovers correctly
    assert all(p.brand_normalized == "pepsi" for p in r2.products)


# --- J: non-regression -- OOC laptop charger / car tires, bag of chips ---

def test_ooc_controls_unaffected():
    from src.scenario.orchestrator import _lacks_catalog_evidence
    for query_text in ("laptop charger", "car tires"):
        assert _lacks_catalog_evidence(FilterSet(query_text=query_text)) is True


def test_bag_of_chips_regression_unaffected():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(Action(intent=Intent.SEARCH, raw_query_text="chips", category_hint="chips"))
    search = RecordingSearch()

    r = handle_message("s1", "i want a bag of chips", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search)

    assert r.likely_out_of_catalog is False
    assert len(search.calls) == 1


# ---------------------------------------------------------------------
# Sprint 6 history-dependent investigation: the ORIGINAL live browser bug
# was a 3-turn conversation, not an isolated query. Two live Groq calls
# (across this task and the prior one, both within their own 1-call
# budgets) proved:
#   - turn 3 IN ISOLATION: understand() confidently normalizes
#     raw_query_text="pepsi", category_hint="soft drink",
#     brand_text="Pepsi".
#   - turn 3 WITH the real preceding history (2 failed "bebsi" turns,
#     confirmed via the real raw history text the orchestrator actually
#     accumulates -- OOC turns DO append to history): understand()
#     instead returns raw_query_text="pebsi" (uncorrected),
#     category_hint=None, brand_text=None, customer_service_tone=True --
#     a real, reproducible history-dependent degradation of the Action.
#
# Despite that degradation, tracing this EXACT degraded Action through
# validate() -> the out-of-catalog gate -> SearchAdapter end-to-end shows
# it already resolves correctly: query_text="pebsi" is a single word (the
# multi-word corpus-cooccurrence gate doesn't apply), so it reaches
# SearchAdapter, whose PRE-EXISTING correct_unmatched_terms() spelling
# corrector (unrelated to the prior task's brand-path fix) recovers
# "pebsi"->"pepsi" and returns real Pepsi products. The original browser
# failure (Bebem/Babyliss) does NOT reproduce against the current
# codebase with either the isolated or the historically-degraded Action
# -- confirmed via the real handle_message() orchestrator, not a
# standalone trace. No production change was needed or made; this locks
# the exact 3-turn sequence in as a regression test.
# ---------------------------------------------------------------------

def test_exact_three_turn_history_no_longer_returns_unrelated_products():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        # Turns 1-2: reconstructed from observed browser symptoms (broad
        # soft-drink results, then out_of_catalog) -- both consistent
        # with "bebsi" (60.0 vs "pepsi") never clearing any threshold.
        Action(intent=Intent.SEARCH, category_hint="soft drink", brand_text="bebsi"),
        Action(intent=Intent.SEARCH, raw_query_text="bebsi dump", brand_text="bebsi"),
        # Turn 3: the REAL, live-confirmed, history-degraded Action.
        Action(intent=Intent.SEARCH, raw_query_text="pebsi", customer_service_tone=True),
    )
    search = RecordingSearch()

    r1 = handle_message("s1", "now i want bottel of bebsi", state_manager=sm, activity_store=store,
                         understand_fn=understand_fn, search_fn=search)
    assert r1.likely_out_of_catalog is False  # broad soft-drink results, not OOC

    r2 = handle_message("s1", "i said bebsi dump", state_manager=sm, activity_store=store,
                         understand_fn=understand_fn, search_fn=search)
    assert r2.likely_out_of_catalog is True  # matches the observed live OOC turn
    assert store.get("s1").history == ["now i want bottel of bebsi", "i said bebsi dump"]

    r3 = handle_message("s1", "how you cant find pebsi", state_manager=sm, activity_store=store,
                         understand_fn=understand_fn, search_fn=search)

    assert r3.likely_out_of_catalog is False
    assert r3.products  # NOT empty, NOT Bebem/Babyliss
    assert all(p.brand_normalized in (None, "pepsi") for p in r3.products)
    assert sum(1 for p in r3.products if "pepsi" in p.name_en.lower()) >= 8


# --- D: a failed weak typo followed by the EXACT (non-typo) brand name ---

def test_failed_typo_then_exact_brand_name_works():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, category_hint="soft drink", brand_text="bebsi"),
        Action(intent=Intent.SEARCH, raw_query_text="pepsi", brand_text="Pepsi", category_hint="soft drink"),
    )
    search = RecordingSearch()

    r1 = handle_message("s1", "bebsi", state_manager=sm, activity_store=store,
                         understand_fn=understand_fn, search_fn=search)
    assert r1.resolved_brand is None

    r2 = handle_message("s1", "pepsi", state_manager=sm, activity_store=store,
                         understand_fn=understand_fn, search_fn=search)
    assert r2.resolved_brand == "pepsi"
    assert all(p.brand_normalized == "pepsi" for p in r2.products)


# --- F: a second real-brand correction-after-failure sequence (not Pepsi) ---

def test_second_brand_correction_after_failure_sequence():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        # "detol" (a weaker, ambiguous typo of "dettol") fails first...
        Action(intent=Intent.SEARCH, raw_query_text="detol", brand_text="detol"),
        # ...then "dettel" (the proven, strongly-evidenced typo) recovers.
        Action(intent=Intent.SEARCH, raw_query_text="dettel", brand_text="dettel"),
    )
    search = RecordingSearch()

    handle_message("s1", "detol", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)

    r2 = handle_message("s1", "dettel", state_manager=sm, activity_store=store,
                         understand_fn=understand_fn, search_fn=search)
    assert r2.resolved_brand == "dettol"
    assert r2.products
    assert all(p.brand_normalized == "dettol" for p in r2.products)
