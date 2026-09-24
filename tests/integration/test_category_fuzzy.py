"""Permanent regression coverage for category AND brand resolution
(English + Arabic) in the keyword-search parser, turning the full
catalog-wide verification sweep (research.md) into automated tests so a
future change can't silently regress it.

Covers, matching what was actually verified against the real catalog and
the real end-to-end parser (parse_keyword_query, the same code path a
real user query hits -- not an isolated internal function):
1. Every real category name AND every real brand name in the catalog, in
   three forms each: (a) correctly spelled alone, (b) with a single
   realistic typo, (c) combined with a real brand/category that actually
   co-occurs with it in the catalog data, in both word orders. Asserts
   each bucket's pass rate doesn't regress below its measured baseline,
   and lists every individual failure by name if it does.
2. The root-cause fixes found via that sweep: a fuzzy-match tie-break bug
   that fragmented multi-word brand names ("Aqua Delta" -> just "Aqua"),
   category and brand competing for the same text instead of one blindly
   running before the other ("Ahmed Tea" losing "Tea" to the category
   matcher), and the SearchAdapter vector-fusion bug that let irrelevant
   same-brand products flood results when free text combined with a
   brand filter ("لانشون اطياب" returning all 71 Atyab products instead
   of the 9 genuine luncheon matches).
3. The confirmed false-positive collisions (a generic/brand word
   coincidentally scoring >= threshold against an unrelated real
   category or brand), each with its own assertion.
4. The specific real queries already discussed, with their exact
   verified expected values.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import psycopg  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.api.routes import app  # noqa: E402
from src.ingestion.text_normalize import is_arabic  # noqa: E402
from src.models.config import settings  # noqa: E402
from src.search_engine.keyword_baseline import (  # noqa: E402
    _REFERENCED_CATEGORIES_SQL,
    parse_keyword_query,
    search_keyword,
)

client = TestClient(app)

LATIN_ALPHABET = "abcdefghijklmnopqrstuvwxyz"
ARABIC_ALPHABET = "ابتثجحخدذرزسشصضطظعغفقكلمنهوي"


def _is_arabic_char(ch: str) -> bool:
    return "؀" <= ch <= "ۿ"


def _make_typo(name: str, seed: str) -> str | None:
    """One single-character substitution typo, deterministic via a STRING
    seed passed straight to random.Random -- not hash(...): Python
    randomizes str hashing per-process, which made an earlier version of
    this sweep silently non-reproducible run to run."""
    lang = "ar" if is_arabic(name) else "en"
    chars = list(name)
    if lang == "ar":
        positions = [i for i, c in enumerate(chars) if _is_arabic_char(c)]
        alphabet = ARABIC_ALPHABET
    else:
        positions = [i for i, c in enumerate(chars) if c.isalpha()]
        alphabet = LATIN_ALPHABET
    if len(positions) < 2:
        return None
    rng = random.Random(seed)
    pos = rng.choice(positions)
    orig = chars[pos]
    repl = rng.choice([c for c in alphabet if c.lower() != orig.lower()])
    out = chars.copy()
    out[pos] = repl
    return "".join(out)


def _load_catalog_and_cooccurrence():
    url = settings.database_url.replace("postgresql+psycopg://", "postgresql://")
    with psycopg.connect(url) as conn, conn.cursor() as cur:
        cur.execute("SELECT slug, display_name_en, display_name_ar FROM canonical_category ORDER BY slug")
        categories = cur.fetchall()

        cur.execute("SELECT slug, display_name FROM canonical_brand ORDER BY slug")
        brands = cur.fetchall()

        cur.execute("""
            SELECT cc.slug, cb.slug, COUNT(*) cnt
            FROM product_category pc
            JOIN category c ON c.id = pc.category_id
            JOIN canonical_category cc ON cc.id = c.canonical_category_id
            JOIN product p ON p.id = pc.product_id
            JOIN canonical_brand cb ON cb.slug = p.brand_normalized
            GROUP BY cc.slug, cb.slug
        """)
        cat_brand_pairs = cur.fetchall()

    best_brand_for_cat: dict[str, tuple[str, int]] = {}
    best_cat_for_brand: dict[str, tuple[str, int]] = {}
    for cat_slug, brand_slug, cnt in cat_brand_pairs:
        if cat_slug not in best_brand_for_cat or cnt > best_brand_for_cat[cat_slug][1]:
            best_brand_for_cat[cat_slug] = (brand_slug, cnt)
        if brand_slug not in best_cat_for_brand or cnt > best_cat_for_brand[brand_slug][1]:
            best_cat_for_brand[brand_slug] = (cat_slug, cnt)

    return categories, brands, best_brand_for_cat, best_cat_for_brand


# Pre-existing, duplicate canonical_category rows sharing the identical
# Arabic display name (a Sprint 0 canonicalization data artifact, not a
# matching-logic bug -- e.g. 'fresh_cheese' and 'cheese_fresh_1751' both
# display "الجبن الطازج"). A "combined" test can legitimately resolve to
# either twin; both are equally correct text matches, so these pairs are
# treated as interchangeable rather than a failure.
_KNOWN_DUPLICATE_CATEGORY_PAIRS = [
    {"fresh_cheese", "cheese_fresh_1751"},
    {"cleaning_products_5", "detergents_19"},
    {"electronics", "electronics_offers_3306"},
    {"meat", "meats_1773"},
    {"nuts", "nuts_seeds_16871"},
    {"sauces_pastes_5694", "sauces_5696"},
]

# Real brand names that are also, character-for-character, a real
# category name ("Beauty", "Corona", "Hero"). Exact category matching
# resolves these before fuzzy matching (or brand) ever runs -- a
# pre-existing, deliberate precedent, not something this sweep tries to
# change.
_KNOWN_EXACT_TIE_BRANDS = {"beauty", "corona", "hero"}


def test_full_catalog_category_and_brand_sweep():
    categories, brands, best_brand_for_cat, best_cat_for_brand = _load_catalog_and_cooccurrence()
    brand_display_by_slug = {s: d for s, d in brands}
    cat_names_by_slug = {s: (en, ar) for s, en, ar in categories}

    buckets: dict[str, list[tuple]] = {
        "cat_correct": [], "cat_typo": [], "cat_combined": [],
        "brand_correct": [], "brand_typo": [], "brand_combined": [],
    }

    def record(bucket, ok, label, query, got_cat, got_brand):
        buckets[bucket].append((ok, label, query, got_cat, got_brand))

    # --- CATEGORY: every real category, EN + AR ---
    for slug, en, ar in categories:
        co_brand_slug = best_brand_for_cat.get(slug, (None, 0))[0]
        for lang, name in (("en", en), ("ar", ar)):
            name = (name or "").strip()
            if not name:
                continue

            fs = parse_keyword_query(name)
            record("cat_correct", fs.category == slug, slug, name, fs.category, fs.brand)

            typo = _make_typo(name, seed=f"{slug}:{lang}:c")
            if typo:
                fs = parse_keyword_query(typo)
                record("cat_typo", fs.category == slug, slug, typo, fs.category, fs.brand)

            if co_brand_slug:
                brand_name = brand_display_by_slug[co_brand_slug]
                for combo in (f"{name} {brand_name}", f"{brand_name} {name}"):
                    fs = parse_keyword_query(combo)
                    cat_ok = fs.category == slug or (
                        fs.category is not None
                        and {fs.category, slug} in _KNOWN_DUPLICATE_CATEGORY_PAIRS
                    )
                    ok = cat_ok and fs.brand == co_brand_slug
                    record("cat_combined", ok, slug, combo, fs.category, fs.brand)

    # --- BRAND: every real canonical brand ---
    for slug, display in brands:
        name = (display or "").strip()
        if not name:
            continue
        co_cat_slug = best_cat_for_brand.get(slug, (None, 0))[0]

        fs = parse_keyword_query(name)
        ok = fs.brand == slug or (slug in _KNOWN_EXACT_TIE_BRANDS and fs.category is not None)
        record("brand_correct", ok, slug, name, fs.category, fs.brand)

        typo = _make_typo(name, seed=f"{slug}:b")
        if typo:
            fs = parse_keyword_query(typo)
            record("brand_typo", fs.brand == slug, slug, typo, fs.category, fs.brand)

        if co_cat_slug:
            cat_en, cat_ar = cat_names_by_slug[co_cat_slug]
            cat_name = cat_en or cat_ar
            for combo in (f"{name} {cat_name}", f"{cat_name} {name}"):
                fs = parse_keyword_query(combo)
                ok = fs.brand == slug and fs.category == co_cat_slug
                record("brand_combined", ok, slug, combo, fs.category, fs.brand)

    # Measured baselines (research.md) -- a drop below any of these means
    # a real regression, not just an imperfect but stable state. cat_typo
    # and brand_typo stay well under 100% for an understood, inherent
    # reason: a single-character substitution in a short (3-4 char) name
    # mathematically drops edit-distance ratio below threshold (same
    # tension already accepted for the fuzzy threshold generally -- see
    # test_known_inherent_limitation_short_word_substitution).
    floors = {
        "cat_correct": 0.98, "cat_typo": 0.78, "cat_combined": 0.97,
        "brand_correct": 0.99, "brand_typo": 0.77, "brand_combined": 0.98,
    }

    report_lines = []
    overall_pass = overall_total = 0
    for bucket, cases in buckets.items():
        total = len(cases)
        passed = sum(1 for c in cases if c[0])
        overall_pass += passed
        overall_total += total
        rate = passed / total if total else 1.0
        report_lines.append(f"{bucket}: {passed}/{total} = {rate:.4f} (floor {floors[bucket]})")
        if rate < floors[bucket]:
            failures = [c for c in cases if not c[0]]
            report_lines.append(f"  FAILURES ({len(failures)}):")
            for ok, label, query, got_cat, got_brand in failures:
                report_lines.append(f"    {label!r}: {query!r} -> category={got_cat!r} brand={got_brand!r}")

    assert overall_total > 5000, f"expected ~5756 generated cases, got {overall_total} -- catalog changed?"

    report = "\n".join(report_lines)
    for bucket, cases in buckets.items():
        total = len(cases)
        passed = sum(1 for c in cases if c[0])
        rate = passed / total if total else 1.0
        assert rate >= floors[bucket], f"{bucket} pass rate regressed to {rate:.4f}\n\nFull report:\n{report}"


def test_known_inherent_limitation_short_word_substitution():
    """Documents, rather than hides, a real and inherent limitation: a
    single-character SUBSTITUTION in a short (3-4 char) name drops the
    edit-distance ratio below threshold for both category and brand
    (e.g. 'meat'->'zeat' scores 75.0, below 80) -- a lower threshold
    would reopen false positives (research.md). Not asserting this
    should ever start passing; asserting it fails in the SAME, understood
    way, so a behavior change here is noticed, not silently absorbed."""
    assert parse_keyword_query("zeat").category is None
    assert parse_keyword_query("rioe").category is None
    assert parse_keyword_query("milz").category is None


# --- Root-cause fixes found via the full sweep --------------------------

def test_fuzzy_tiebreak_prefers_longer_more_specific_brand_match():
    # Real bug: a 2-word brand ('Aqua Delta') was getting chopped down to
    # a DIFFERENT, shorter, also-real brand ('Aqua') because both scored
    # a tied 100.0 and the shorter window was generated (and kept) first.
    for full_name, expected_slug in [
        ("Aqua Delta", "aqua_delta"),
        ("Big Babol", "big_babol"),
        ("Dabur Amla", "dabur_amla"),
        ("Fresh Days", "fresh_days"),
    ]:
        fs = parse_keyword_query(full_name)
        assert fs.brand == expected_slug, f"{full_name!r} -> {fs.brand!r}, expected {expected_slug!r}"


def test_category_and_brand_resolved_as_competing_matches_not_sequential():
    # Real bug: a real multi-word brand name containing a real category
    # word ('Ahmed Tea', 'My Meat', 'Coffee Break') lost that word to
    # category matching, which ran unconditionally first and had no way
    # to know a longer, more specific brand match existed for the same
    # span. Category and brand now compete over the same text, and
    # whichever match is longer/more specific wins.
    fs = parse_keyword_query("Ahmed Tea")
    assert fs.brand == "ahmed_tea"
    assert fs.category is None

    fs = parse_keyword_query("My Meat")
    assert fs.brand == "my_meat"
    assert fs.category is None

    fs = parse_keyword_query("Coffee Break")
    assert fs.brand == "coffee_break"
    assert fs.category is None


def test_searchadapter_does_not_flood_results_with_irrelevant_same_brand_products():
    # The actual root cause of the reported "لانشون اطياب" bug: category
    # correctly resolves to None and brand correctly resolves to 'atyab'
    # (no typo, no category-matching bug at all) -- the bug was in
    # SearchAdapter unconditionally fusing in vector_rank's results, which
    # has no relevance floor and returned literally all 71 Atyab products
    # regardless of relevance, instead of the 9 genuine BM25 matches for
    # "لانشون" (luncheon). Verified against the real product catalog: all
    # 71 Atyab products would previously appear; only luncheon ones should.
    r = search_keyword("لانشون اطياب", limit=20)
    assert r.total_count == 9, f"expected exactly the 9 genuine luncheon matches, got {r.total_count}"
    for p in r.products:
        assert p["brand_normalized"] == "atyab"
        assert "luncheon" in p["name_en"].lower()

    # The typo/cross-lingual rescue path (BM25 finds nothing) must still
    # work -- this is what the vector fallback exists for.
    r2 = search_keyword("milq", limit=10)
    assert not r2.zero_result
    assert any("milk" in p["name_en"].lower() for p in r2.products)


def test_generic_word_typo_no_longer_hard_filters_to_unrelated_brand():
    # Real live bug: "لين" (typo of "لبن"/milk) scored 85.7 against the
    # real brand "لينو" (Lino) and hard-filtered results to Lino-only
    # products, excluding milk entirely. Brand must stay unmatched for a
    # word that's really just a typo of a generic term, not a brand
    # mention -- confirmed via a broader sweep of common Arabic/English
    # words (not a one-off patch for this single word).
    #
    # "cookie" (vs real brand "Cooker", 83.3) and "dish" (vs real brand
    # "Modish", 80.0) are the same class of bug, found live via the
    # /search/keyword endpoint: q="cookie" returned 7 Cooker-brand
    # canned/pantry products and ZERO real cookies before this fix.
    # Neither "cookie" nor "dish" is itself a real raw brand alias or
    # canonical brand (confirmed against the live table).
    for query in ["لين", "شامبو", "dairy", "clean", "spicy", "cookie", "dish"]:
        fs = parse_keyword_query(query)
        assert fs.brand is None, f"{query!r} -> brand={fs.brand!r}, expected None"


def test_real_brand_cooker_and_modish_still_resolve_explicitly():
    """Blocklisting the coincidental collision words "cookie"/"dish" must
    not block the real, distinct brands they collide with -- an explicit
    mention of either brand must still resolve."""
    assert parse_keyword_query("cooker").brand == "cooker"
    assert parse_keyword_query("modish").brand == "modish"
    assert parse_keyword_query("chicken cooker").brand == "cooker"
    assert parse_keyword_query("modish hair gel").brand == "modish"


def test_filler_stopwords_never_resolve_to_an_unrelated_brand():
    # Confirmed via a systematic sweep of ~144 common EN/AR filler/
    # stopwords through the real resolver, not guessed one at a time
    # (research.md): 33 of them scored >= threshold against a real,
    # unrelated brand -- e.g. "or" vs "Ors" (80.0), "breast" (from
    # "chiken breast") vs "Beast" (90.9), "براند" (Arabic for "brand",
    # used generically) vs "براون"/Braun (80.0). "sure"/"fine"/"today"/
    # "one" ARE literally real brand names (100.0) -- still blocked here
    # since they're common enough words that a bare mention is far more
    # likely to be filler than an actual brand search.
    words = [
        "an", "or", "for", "any", "sure", "fine", "today", "in", "on", "at",
        "much", "but", "if", "give", "shall", "down", "off", "over", "again",
        "one", "breast",
        "في", "إلى", "الى", "هل", "لا", "كام", "لأ", "انا", "انتي", "فيه",
        "فين", "اي", "براند",
    ]
    for query in words:
        fs = parse_keyword_query(query)
        assert fs.brand is None, f"{query!r} -> brand={fs.brand!r}, expected None"


# --- Confirmed false-positive collisions (category vs. brand/word) ------

@pytest.mark.parametrize("query", [
    "price 100",       # vs real category "Rice", scores 88.9
    "venus",           # vs "Ovens", 80.0
    "cheesa",          # vs "Cheese", 83.3
    "break",           # vs "Bread", 80.0
    "dream",           # vs "Cream", 80.0
    "delia",           # vs "Deli", 88.9
    "milka",           # vs "Milk", 88.9 -- also stole a real brand match
    "astra",           # vs "Pasta", 80.0
    "beast",           # vs "Beans", 80.0
    "chocodate",       # vs "Chocolate", 88.9
    "nucream",         # vs "Cream", 83.3
    # Found via the same systematic filler/stopword sweep as the brand
    # blocklist (test_filler_stopwords_never_resolve_to_an_unrelated_
    # brand): "me" vs real category "Men" (80.0), and three Arabic
    # prepositions/pronouns that only cross threshold with the "ال"-
    # stripping tolerance applied: "من" (from) vs "السمن"/ghee (80.0),
    # "زي" (like) vs "الزيت"/oil (80.0), "اي" (any/which) vs "الشاي"/tea
    # (80.0) -- "اي" also needed a SECOND fix in the brand blocklist after
    # this one, since it fell through to a different real brand ("كاى").
    "me", "من", "زي", "اي",
])
def test_category_fuzzy_blocklisted_false_positives_stay_unmatched(query):
    assert parse_keyword_query(query).category is None


@pytest.mark.parametrize("query,expected_slug", [
    ("جبنة", "cheese"),            # singular "a cheese" vs plural "الأجبان" (54.5)
    ("جبن", "cheese"),             # mass-noun "cheese" vs plural "الأجبان"
    ("لحمة", "meat"),              # singular "a meat" vs plural "اللحوم" (60.0)
    ("لبن", "milk_29"),            # generic "milk" vs plural "الألبان" (60.0)
    ("حليب", "milk_29"),           # alt word "milk" vs plural "الألبان"
    ("سمك", "seafood"),            # mass-noun "fish" vs plural "أسماك" (75.0)
    ("سمكة", "seafood"),           # singular "a fish" vs plural "أسماك"
    ("خضار", "vegetables_3286"),   # mass-noun "vegetables" vs "خضراوات" (54.5)
    ("خضرة", "vegetables_3286"),   # singular-ish "a vegetable" vs "خضراوات"
    ("مكسرة", "nuts_seeds_16871"),  # singular "a nut" vs plural "المكسرات" (61.5)
    ("حبة", "cereals_5700"),       # singular "a grain" vs plural "الحبوب" (44.4)
    ("خضار معلبة", "canned_vegetables_fruits_5697"),  # compound, vs "خضراوات معلبة" (78.3)
    ("خضار مجمدة", "frozen_vegetables_11674"),        # compound, vs "خضراوات مجمدة" (78.3)
])
def test_category_singular_form_aliases_resolve_correctly(query, expected_slug):
    # Confirmed via a systematic sweep of all 244 real categories, not
    # just the 2 originally-known cases (research.md Class B): a real
    # customer's natural singular/mass-noun form of a countable grocery
    # item often falls short of the stored plural/collective category
    # name's fuzzy threshold. Real scores in comments above. Fixed via
    # explicit per-category singular-form aliases
    # (_CATEGORY_SINGULAR_ALIASES), not a general Arabic pluralization
    # rule -- Arabic "broken plurals" don't follow a simple suffix pattern
    # (جبنة -> أجبان isn't a suffix change).
    assert parse_keyword_query(query).category == expected_slug


def test_category_singular_alias_does_not_shadow_a_more_specific_compound_match():
    # Real regression found while adding the aliases above: "لحمة" (alias
    # for 'meat') is a perfect 1-word match, so it always won immediately
    # over the real, longer "لحوم معلبة"/"لحوم مجمدة" compound match
    # (90.0) -- both because exact matching short-circuits fuzzy
    # matching entirely, and because within fuzzy matching itself the
    # alias's own 100.0 self-match outscores the compound's 90.0 before
    # length is ever compared. _resolve_category() now re-checks fuzzy
    # matching against the real category names (excluding the alias
    # phrases themselves) whenever the exact match came from an alias,
    # and prefers a longer result.
    fs = parse_keyword_query("لحمة معلبة")
    assert fs.category == "canned_meat_25415"
    fs2 = parse_keyword_query("لحمة مجمدة")
    assert fs2.category in ("frozen_meat", "frozen_beef_10484")  # duplicate-name twins, both correct

    # The bare alias, and a genuinely longer/more-specific EXISTING match,
    # must still both work correctly (not regressed by the fix above).
    assert parse_keyword_query("لحمة").category == "meat"
    assert parse_keyword_query("لبن كامل الدسم").category == "full_cream_milk_31"


def test_category_fuzzy_blocklisted_two_word_false_positive():
    # "Farm Cheese" (real brand) vs "Romy cheese" (real category) scores
    # 81.8 -- blocklisted as a 2-word category-fuzzy phrase. Since "Farm
    # Cheese" is ALSO the real, correct, more specific 2-word brand name,
    # joint resolution now correctly prefers that longer brand match over
    # the shorter "cheese" category match that used to win here.
    fs = parse_keyword_query("farm cheese")
    assert fs.brand == "farm_cheese"
    assert fs.category is None


def test_category_fuzzy_does_not_steal_a_brand_match():
    fs = parse_keyword_query("milka")
    assert fs.category is None
    assert fs.brand == "milka"
    fs2 = parse_keyword_query("milka chocolate")
    assert fs2.category == "chocolate_2"
    assert fs2.brand == "milka"


# --- Specific real cases, exact expected values -------------------------

def test_real_case_boskot_typo_resolves_to_biscuits():
    fs = parse_keyword_query("بسكوت")
    assert fs.category == "biscuits_3"
    assert fs.brand is None
    assert fs.price_max is None


def test_real_case_boskot_typo_with_price_resolves_to_biscuits():
    fs = parse_keyword_query("بسكوت تحت 30")
    assert fs.category == "biscuits_3"
    assert fs.price_max == 30.0


def test_real_case_original_juhayna_full_cream_milk_query():
    fs = parse_keyword_query("لين جهينه كامل الدسم تحت 55")
    assert fs.category == "full_cream_milk_31"
    assert fs.brand == "juhayna"
    assert fs.price_min is None
    assert fs.price_max == 55.0


def test_real_case_laban_juhayna_no_category_word():
    # "لبن" alone used to leave category unresolved (no category is
    # literally "لبن" alone in this catalog) -- now resolves via its
    # singular-form alias for milk_29 (research.md Class B fix), and
    # brand still resolves independently.
    fs = parse_keyword_query("لبن جهينه تحت 50")
    assert fs.category == "milk_29"
    assert fs.brand == "juhayna"
    assert fs.price_max == 50.0


def test_real_case_haleeb_juhayna_alternate_spelling():
    # 'حليب' (milk, alternate word) now also resolves via its singular-
    # form alias for milk_29 (research.md Class B fix). 'جهينة' (with taa
    # marbuta) is a spelling variant of the stored alias 'جهينه', still
    # resolved via fuzzy matching.
    fs = parse_keyword_query("حليب جهينة تحت 55")
    assert fs.category == "milk_29"
    assert fs.brand == "juhayna"
    assert fs.price_max == 55.0


def test_real_case_laban_full_cream_no_typo_unaffected():
    fs = parse_keyword_query("لبن كامل الدسم")
    assert fs.category == "full_cream_milk_31"
    assert fs.brand is None


def test_real_case_atyab_luncheon_via_http_endpoint():
    r = client.get("/search/keyword", params={"q": "لانشون اطياب"})
    assert r.status_code == 200
    data = r.json()
    assert data["resolved_brand"] == "atyab"
    assert data["total_count"] == 9


def test_orphaned_canonical_category_row_is_excluded_from_resolution_candidates():
    """category_canonicalization_apply.py resets category.canonical_category_id
    on every re-apply but never deletes a canonical_category row a changed
    mapping stops producing (independent audit finding D) --
    _get_catalog_names() now filters to referenced-only rows
    (_REFERENCED_CATEGORIES_SQL) so such an orphan is harmless to live
    resolution. Proven directly against the real DB inside a transaction
    guaranteed to roll back -- no canonical_category row is left behind."""
    url = settings.database_url.replace("postgresql+psycopg://", "postgresql://")
    conn = psycopg.connect(url)
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO canonical_category (slug, display_name_en, display_name_ar) "
            "VALUES (%s, %s, %s)",
            ("orphan_test_slug_zzz", "Orphan Test", "تجربة يتيمة"),
        )

        cur.execute(_REFERENCED_CATEGORIES_SQL)
        slugs = {row[0] for row in cur.fetchall()}

        assert "orphan_test_slug_zzz" not in slugs, "orphaned row must not be a resolution candidate"
        assert "milk_29" in slugs, "a real, referenced category must still be a candidate"
    finally:
        conn.rollback()
        conn.close()

    # Confirm the rollback actually took effect -- the test row must not exist.
    with psycopg.connect(url) as verify_conn, verify_conn.cursor() as verify_cur:
        verify_cur.execute("SELECT count(*) FROM canonical_category WHERE slug = %s",
                            ("orphan_test_slug_zzz",))
        assert verify_cur.fetchone()[0] == 0


def test_real_case_cookie_no_longer_hijacked_to_cooker_brand_via_http_endpoint():
    # Confirmed live bug: q="cookie" used to resolve brand="cooker" and
    # return 7 Cooker-brand canned/pantry products (Altahya fava beans,
    # corn, mushrooms, rice), zero real cookies. Real corpus count
    # right after the brand fix (2026-09): 47 results. Rose to 102 once
    # the tokenizer's "-ies" ambiguity was ALSO fixed (research.md) --
    # "cookie" and "cookies" now stem to the same token ("cookie", not the
    # old "cooky"), so BM25 literal matching finds the full combined
    # candidate set instead of only the singular-token subset.
    r = client.get("/search/keyword", params={"q": "cookie"})
    assert r.status_code == 200
    data = r.json()
    assert data["resolved_brand"] is None
    assert data["total_count"] == 102
    names = [p["name_en"].lower() for p in data["products"]]
    assert all(p["brand_normalized"] != "cooker" for p in data["products"])
    assert any("cookie" in n for n in names)


def test_real_case_dish_no_longer_restricted_to_modish_brand_via_http_endpoint():
    # Confirmed live bug: q="dish" used to resolve brand="modish" (a
    # hair-care brand), restricting results to Modish products only.
    # Real corpus count right after the brand fix (2026-09): 112 results.
    # Rose to 117 once the tokenizer's "-es" ambiguity was ALSO fixed
    # (research.md) -- "dish" and "dishes" now both stem to "dish" (not
    # the old "dishe" for the plural), so BM25 finds the full combined set.
    # Dropped to 77 (Sprint 6 search+recommendation context task):
    # vector_rank's fusion contribution is now gated by the SAME
    # VECTOR_FALLBACK_MAX_DISTANCE floor as the vector-only fallback --
    # the 40 removed rows (pasta, sandwiches, cheese spreads, salad
    # dressing, smoked turkey breast) were never genuinely "dish"-related,
    # just semantically-nearby noise fusion previously added unfiltered.
    # 77 is exactly BM25's own literal "dish"/"dishes" count -- both
    # original protections (no brand hijack, no plural-tokenizer split)
    # remain intact; only the never-audited vector-fusion noise is gone.
    r = client.get("/search/keyword", params={"q": "dish"})
    assert r.status_code == 200
    data = r.json()
    assert data["resolved_brand"] is None
    assert data["total_count"] == 77
    assert not all(p["brand_normalized"] == "modish" for p in data["products"])
