"""Sprint 6 out-of-catalog grounding follow-up: strengthens
resolve_category_hint()/resolve_brand_text()'s fuzzy matching
(src/search_engine/keyword_baseline.py's _fuzzy_phrase_match) so an
unattested word coincidentally resembling a real category/brand name can
no longer manufacture false catalog membership -- while preserving every
genuine, already-relied-on typo case. Hermetic in the sense that no LLM
call is involved; these call the real resolvers against the real DB-backed
catalog, same as tests/integration/test_category_fuzzy.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.search_engine.keyword_baseline import resolve_brand_text, resolve_category_hint  # noqa: E402


# --- A: the real regression -- must NOT resolve to an unrelated category ---

def test_mobile_phone_does_not_resolve_to_honey_or_anything_else():
    assert resolve_category_hint("mobile phone") is None


# --- B: unseen unrelated concepts must not fuzzy-resolve either ---

def test_unseen_unrelated_concepts_do_not_resolve():
    for hint in ["guitar strings", "car battery", "printer ink", "yoga mat", "dog leash", "garden hose"]:
        assert resolve_category_hint(hint) is None, hint


# --- C: genuine misspelled real categories/brands must still resolve ---

def test_genuine_category_typos_still_resolve():
    assert resolve_category_hint("بسكوت") == "biscuits_3"
    assert resolve_category_hint("choclate") == "chocolate_2"


def test_genuine_brand_typos_still_resolve():
    assert resolve_brand_text("Juhyna") == "juhayna"
    assert resolve_brand_text("Nestlle") == "nestle"
    assert resolve_brand_text("Pepsii") == "pepsi"
    assert resolve_brand_text("Colgat") == "colgate"


# --- D: exact category still resolves ---

def test_exact_category_still_resolves():
    assert resolve_category_hint("rice") == "rice"
    assert resolve_category_hint("milk") == "milk_29"


# --- E: Arabic real category still resolves ---

def test_arabic_real_category_still_resolves():
    assert resolve_category_hint("جبنة") == "cheese"


# --- F: code-switch / cross-lingual real brand still resolves ---

def test_arabic_brand_spelling_variant_still_resolves():
    assert resolve_brand_text("جهينه") == "juhayna"


# --- known false-hijack guardrail (pre-existing, must remain fixed) ---

def test_generic_arabic_word_still_does_not_hijack_unrelated_brand():
    assert resolve_brand_text("الوادي") is None


# --- H: laptop charger against the category resolver specifically ---

def test_laptop_charger_does_not_resolve_a_category():
    assert resolve_category_hint("laptop charger") is None
