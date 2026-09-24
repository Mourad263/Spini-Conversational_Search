"""System A: the traditional keyword-search baseline (FR-014, tasks.md T048).

Takes a plain typed string -- no structured FilterSet handed in -- and does
NAIVE literal/regex parsing to pull out a price range and an exact category
name mention if present, then calls the same SearchAdapter every other
system uses. Deliberately dumb: recognizes a fixed set of literal price
phrases in both English ("under 50", "between 20 and 40") and Arabic
("تحت 55", "بين 20 و 40") -- this is still literal phrase matching, not
query understanding (e.g. a typo'd price word, or a phrasing not in the
fixed list, is not recognized). Category name matching (English and
Arabic) is literal substring/token matching too. Brand is resolved purely
from q's free text as well -- same as category/price, there is no
dedicated brand= parameter -- but via fuzzy/typo-tolerant matching
(rapidfuzz) against Sprint 0's canonicalized brand list rather than exact
substring matching, since a real misspelled brand mention ("glaxy",
"munshi") is common and a hard substring match would miss it entirely.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import psycopg  # noqa: E402
from rapidfuzz import fuzz  # noqa: E402

from src.ingestion.text_normalize import is_arabic, normalize_arabic, normalize_latin  # noqa: E402
from src.models.config import settings  # noqa: E402
from src.search_adapter.adapter import FilterSet, SearchAdapter, SearchResult  # noqa: E402
from src.search_engine.index import get_index  # noqa: E402
from src.search_engine.ranking import corpus_cooccurrence_ids, correct_unmatched_terms  # noqa: E402

_MAX_PATTERNS = [
    re.compile(r"\bunder\s+(\d+(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"\bbelow\s+(\d+(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"\bless\s+than\s+(\d+(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"\bcheaper\s+than\s+(\d+(?:\.\d+)?)", re.IGNORECASE),
    # Arabic equivalents -- found missing entirely via a real query
    # ("تحت 55" wasn't parsed at all). [أا] covers both the hamzated
    # (أقل) and the commonly-typed unhamzated (اقل) spelling.
    re.compile(r"\bتحت\s*(\d+(?:\.\d+)?)"),
    re.compile(r"\b[أا]قل\s+من\s*(\d+(?:\.\d+)?)"),
    re.compile(r"\b[أا]رخص\s+من\s*(\d+(?:\.\d+)?)"),
]
_MIN_PATTERNS = [
    re.compile(r"\bover\s+(\d+(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"\babove\s+(\d+(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"\bmore\s+than\s+(\d+(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r"\bفوق\s*(\d+(?:\.\d+)?)"),
    re.compile(r"\b[أا]كثر\s+من\s*(\d+(?:\.\d+)?)"),
]
_BETWEEN_RE = re.compile(r"\bbetween\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)", re.IGNORECASE)
_BETWEEN_AR_RE = re.compile(r"\bبين\s*(\d+(?:\.\d+)?)\s*و\s*(\d+(?:\.\d+)?)")
_RANGE_AR_RE = re.compile(r"\bمن\s*(\d+(?:\.\d+)?)\s*(?:إلى|الى)\s*(\d+(?:\.\d+)?)")
_RANGE_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\b")


def extract_price_constraints(text: str) -> tuple[float | None, float | None, str]:
    """Returns (price_min, price_max, text-with-matched-phrase-removed)."""
    remaining = text
    price_min: float | None = None
    price_max: float | None = None

    m = (_BETWEEN_RE.search(remaining) or _RANGE_RE.search(remaining)
         or _BETWEEN_AR_RE.search(remaining) or _RANGE_AR_RE.search(remaining))
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        price_min, price_max = min(a, b), max(a, b)
        remaining = remaining[:m.start()] + remaining[m.end():]

    if price_max is None:
        for pattern in _MAX_PATTERNS:
            m = pattern.search(remaining)
            if m:
                price_max = float(m.group(1))
                remaining = remaining[:m.start()] + remaining[m.end():]
                break

    if price_min is None:
        for pattern in _MIN_PATTERNS:
            m = pattern.search(remaining)
            if m:
                price_min = float(m.group(1))
                remaining = remaining[:m.start()] + remaining[m.end():]
                break

    return price_min, price_max, remaining


def _normalize_for_fuzzy(text: str) -> str:
    """Same normalization Sprint 0's canonicalization already applies
    (diacritics/alef-variants/case/whitespace) -- used here so fuzzy
    matching measures real spelling differences, not incidental ones."""
    return normalize_arabic(text) if is_arabic(text) else normalize_latin(text)


# Sprint 6 (revised after a catalog-scale audit, not just the original 4
# examples): minimum number of real products that must contain ALL of a
# candidate window's words (see _corpus_cooccurrence_support) before that
# window counts as having "strong real product-corpus evidence" and gets
# HELD to the overlap-ratio check below (_MIN_CANDIDATE_OVERLAP_RATIO) --
# it does not, by itself, decide anything; a well-attested window is
# still accepted if its overlap ratio is high (a genuine variant), only
# rejected if BOTH conditions hold.
#
# A systematic sweep of ~4,400 real fuzzy brand/category collision pairs
# generated from the live catalog (not hand-picked) proved the original
# value of 10 was too permissive: sweeping this cutoff over the pairs
# with overlap_ratio < _MIN_CANDIDATE_OVERLAP_RATIO (the ones this rule
# is actually meant to catch) showed false-hijack count falling from 506
# (at 10) to just 1 (at 3), with ZERO change in how many genuine matches
# were caught either way -- the genuine matches in that same low-overlap
# slice all scored >=95 on their own regardless of the cutoff, so nothing
# is lost by lowering it. 3 is the lowest value directly measured (the
# dataset itself only sampled words with >=3 real occurrences, so a value
# below 3 has no direct evidence either way) and matches this project's
# existing "at least 3 products" convention elsewhere. Confirmed still
# safe for the two closest known near-miss cases: "بسكوت" (a genuine
# colloquial spelling of Biscuits, 3 occurrences) has overlap_ratio=0.667
# against its true category, and "choclate" (a misspelling of chocolate,
# 5 occurrences) has overlap_ratio=0.2 -- both comfortably above
# _MIN_CANDIDATE_OVERLAP_RATIO regardless of this cutoff, so lowering it
# does not newly protect either of them from their own genuine match.
_MIN_PRODUCT_COOCCURRENCE_SUPPORT = 3

# Sprint 6: minimum fraction of a window's own real-product support that
# must ALSO belong to the fuzzy-matched candidate's real product set
# (_corpus_overlap_ratio) before the match is treated as a genuine
# spelling/grammar variant of that SAME entity, exempting it from the
# _MIN_PRODUCT_COOCCURRENCE_SUPPORT protection above. Needed because edit-
# distance score alone cannot separate a genuine variant from a
# coincidence -- they can have the IDENTICAL shape ("جهينة" vs the real
# brand alias "جهينه", and "paste" vs the real category "Pasta", are both
# same-length, single-character-different, both score exactly 80.0).
# Measured directly against real catalog membership (not another string
# guess): the 3 confirmed coincidences (paste/Pasta, peas/Pears,
# frozen+peas/Frozen Meat) all have overlap_ratio EXACTLY 0.0 -- none of
# their real supporting products belong to the colliding candidate at
# all -- while the 3 confirmed genuine variants (جهينة/juhayna,
# detergent/Detergents, كامل+الدسم/Full Cream Milk) score 0.43-0.90. A
# large, clean gap; 0.1 sits safely in the middle of it.
#
# Confirmed at catalog scale, not just these 6 examples: a systematic
# sweep of ~4,400 real collision pairs (67% of them false collisions,
# median overlap_ratio 0.000, max 0.019; genuine matches median 0.895,
# only 4.5% below 0.1) found this cutoff sits in a completely flat,
# consequence-free plateau -- sweeping it across 0.02/0.05/0.1/0.15/0.2/
# 0.3 (holding support-protected pairs fixed) produced the IDENTICAL
# precision/recall every time, meaning no real pair in this catalog
# actually lands between the false-collision ceiling (~0.02) and the
# genuine-match floor (~0.3) -- 0.1 is not a fragile guess sitting on a
# boundary, it has real room on both sides.
_MIN_CANDIDATE_OVERLAP_RATIO = 0.1

# Sprint 6 out-of-catalog grounding follow-up: the score a window with ZERO
# real corpus occurrence of its own must clear before its match is trusted
# (see _fuzzy_phrase_match) -- higher than _CATEGORY_FUZZY_THRESHOLD/
# _BRAND_FUZZY_THRESHOLD (80), but NOT a blind guess. A pure edit-distance
# cutoff was tried first and rejected: standard Levenshtein counts an
# ordinary adjacent-letter transposition ("chocoalte"/chocolate,
# "spinnies"/spinneys, "لحمة معلبة"/"لحوم معلبة" -- all genuine, all
# distance 2) the SAME as the confirmed bug ("phone"/"Honey", also distance
# 2), so distance alone can't separate them. Their fuzz.ratio SCORES do
# separate cleanly, though: every one of those genuine unattested cases
# scores >=87.5, while the confirmed bug scores exactly 80.0 -- a real,
# measured gap. This constant sits just above the confirmed-bug ceiling
# (not near the genuine floor), the same "err toward keeping real matches"
# convention vector_search.py's own VECTOR_FALLBACK_MAX_DISTANCE already
# uses: it excludes the specific coincidences found so far while giving
# every currently-known genuine case a wide (5.5+ point) safety margin,
# at the disclosed cost of not catching a hypothetical future coincidence
# that happens to score 82-87.
_UNATTESTED_WINDOW_MIN_SCORE = 82


def _corpus_cooccurrence_ids(words: list[str]) -> set[int]:
    """Real product IDs whose (name_en + name_ar) tokenization contains
    ALL of `words` together. Thin wrapper over
    src/search_engine/ranking.py's corpus_cooccurrence_ids() -- the ONE
    shared implementation of this postings-intersection primitive, also
    used directly by src/search_adapter/adapter.py's fusion-skip check.
    Kept here, under this exact name, purely so this module's existing
    callers (this module's own fuzzy grounding, src/scenario/
    orchestrator.py's out-of-catalog gate, src/recommendation/service.py)
    need no changes."""
    return corpus_cooccurrence_ids(get_index(), words)


def _corpus_cooccurrence_support(words: list[str]) -> int:
    return len(_corpus_cooccurrence_ids(words))


def _corpus_overlap_ratio(window_ids: set[int], candidate_product_ids: set[int] | None) -> float:
    """Fraction of `window_ids` that are ALSO real instances of the
    candidate brand/category (see the product_ids_by_brand_slug/
    product_ids_by_category_slug catalog maps). 0.0 if the candidate has
    no known product-membership data (never exempts in that case, the
    conservative default) or no overlap at all."""
    if not window_ids or not candidate_product_ids:
        return 0.0
    return len(window_ids & candidate_product_ids) / len(window_ids)


_catalog_cache: dict | None = None

# Arabic natural singular/mass-noun forms a real customer says that the
# canonical category name stores as a plural or collective/definite form
# far enough away (in edit-distance terms) that fuzzy matching alone can't
# bridge it -- e.g. "جبنة" (a cheese) vs the stored "الأجبان" (the
# cheeses) scores only 54.5, well under threshold. Confirmed via a real
# sweep of all 244 categories (research.md), not guessed, and not a
# general Arabic pluralization rule: Arabic "broken plurals" don't follow
# a simple suffix pattern (جبنة -> أجبان isn't a suffix change), and only
# these 9 categories were actually confirmed affected. Each alias is
# added as a real candidate phrase (exact match, and available to fuzzy
# fallback the same as any category name) -- not a separate mechanism.
_CATEGORY_SINGULAR_ALIASES: dict[str, list[str]] = {
    "cheese": ["جبنة", "جبن"],
    "meat": ["لحمة"],
    "milk_29": ["لبن", "حليب"],
    "seafood": ["سمك", "سمكة"],
    "vegetables_3286": ["خضار", "خضرة"],
    "nuts_seeds_16871": ["مكسرة"],
    "cereals_5700": ["حبة"],
    "canned_vegetables_fruits_5697": ["خضار معلبة"],
    "frozen_vegetables_11674": ["خضار مجمدة"],
}


# Sprint 5 Part 3: evidence-backed COMPOUND phrase aliases -- unlike
# _CATEGORY_SINGULAR_ALIASES above (a singular-vs-plural Arabic noun-form
# gap), these cover a different real gap: "full fat"/"low fat" are
# genuine dairy fat-content descriptors used in real product titles, but
# the canonical category names themselves use different terminology
# ("Full Cream Milk"/"Skimmed Milk", not "Full Fat"/"Low Fat"). Confirmed
# via direct corpus lookup (not guessed): every real product whose title
# contains the phrase "Full Fat Milk" (14 distinct products) is tagged
# with full_cream_milk_31 whenever that specific category appears in its
# multi-category list; the only 2 real products titled "... Low Fat ..."
# that are ALSO in a milk-family category are both tagged skimmed_milk_39.
# Deliberately scoped to the full 2-word-plus-"milk" phrase (not a bare
# "full fat"/"low fat" alias) -- both terms are ALSO used generically
# across cheese/meat product titles in this same catalog (confirmed via
# the same lookup: "Low Fat White Cheddar Cheese", "Minced Beef Low Fat"),
# so a bare alias would wrongly redirect an unrelated cheese/meat query to
# a milk category. No equivalent evidence was found for the Arabic "قليل
# الدسم" ("little fat") phrasing applied to milk specifically (every real
# product using that exact Arabic phrase in this catalog is a CHEESE
# product, zero are milk) -- left unresolved rather than guessed; "لبن
# كامل الدسم" needs no alias since it's already the category's own exact
# stored Arabic name.
_CATEGORY_COMPOUND_ALIASES: dict[str, list[str]] = {
    "full_cream_milk_31": ["full fat milk"],
    "skimmed_milk_39": ["low fat milk"],
}


_REFERENCED_CATEGORIES_SQL = """
    SELECT DISTINCT cc.slug, cc.display_name_en, cc.display_name_ar
    FROM canonical_category cc
    JOIN category c ON c.canonical_category_id = cc.id
"""
# Only categories referenced by at least one real `category` row are exposed
# as resolution candidates -- category_canonicalization_apply.py resets
# category.canonical_category_id on every re-apply but never deletes a
# canonical_category row that a CHANGED mapping stops producing, so an
# unfiltered `SELECT * FROM canonical_category` can return a stale, orphaned
# row that would resolve to zero real products (confirmed via a ROLLBACK-only
# demo: an orphaned row was still returned by that unfiltered query). The
# JOIN makes such a row harmless to live resolution without deleting it or
# changing any existing canonical_category id/slug.


def _get_catalog_names() -> dict:
    """Lazily cached: canonical category/brand names to literally match
    against. Same singleton-on-first-use pattern as SearchIndex."""
    global _catalog_cache
    if _catalog_cache is not None:
        return _catalog_cache

    url = settings.database_url.replace("postgresql+psycopg://", "postgresql://")
    with psycopg.connect(url) as conn, conn.cursor() as cur:
        cur.execute(_REFERENCED_CATEGORIES_SQL)
        categories = [{"slug": s, "en": en, "ar": ar} for s, en, ar in cur.fetchall()]

        cur.execute("SELECT raw_value, canonical_brand_id FROM brand_alias")
        alias_rows = cur.fetchall()
        cur.execute("SELECT id, slug FROM canonical_brand")
        brand_slug_by_id = dict(cur.fetchall())
        brands = [
            {"raw": raw, "slug": brand_slug_by_id[cbid], "norm": _normalize_for_fuzzy(raw)}
            for raw, cbid in alias_rows if cbid in brand_slug_by_id
        ]

        # Sprint 6: real product_id -> brand/category slug membership,
        # used ONLY by _corpus_overlap_ratio to tell a genuine spelling/
        # grammar variant of a real entity ("جهينة" for the real brand
        # "جهينه", "detergent" for the real category "Detergents") apart
        # from a merely-coincidental fuzzy resemblance to an UNRELATED
        # entity ("paste" for "Pasta", "peas" for "Pears") -- edit-distance
        # score alone cannot tell these apart (paste/pasta and جهينة/جهينه
        # are BOTH single-character-different, same length, same shape),
        # so this reuses the authoritative DB-side product->brand/category
        # membership directly, not another string-similarity guess.
        product_ids_by_brand_slug: dict[str, set[int]] = {}
        cur.execute("SELECT id, brand_normalized FROM product WHERE brand_normalized IS NOT NULL")
        for pid, slug in cur.fetchall():
            product_ids_by_brand_slug.setdefault(slug, set()).add(pid)

        product_ids_by_category_slug: dict[str, set[int]] = {}
        cur.execute("""
            SELECT pcx.product_id, cc.slug
            FROM product_category pcx
            JOIN category c ON c.id = pcx.category_id
            JOIN canonical_category cc ON cc.id = c.canonical_category_id
        """)
        for pid, slug in cur.fetchall():
            product_ids_by_category_slug.setdefault(slug, set()).add(pid)

    # (slug, phrase, normalized_phrase) -- normalized once here, not on
    # every parse_keyword_query call, since fuzzy category matching scores
    # every candidate against every query word-window.
    category_candidates = [
        (cat["slug"], name, _normalize_for_fuzzy(name))
        for cat in categories for name in (cat["en"], cat["ar"]) if name
    ]
    category_candidates += [
        (slug, alias, _normalize_for_fuzzy(alias))
        for slug, aliases in _CATEGORY_SINGULAR_ALIASES.items() for alias in aliases
    ]
    category_candidates += [
        (slug, alias, _normalize_for_fuzzy(alias))
        for slug, aliases in _CATEGORY_COMPOUND_ALIASES.items() for alias in aliases
    ]

    _catalog_cache = {"categories": categories, "brands": brands,
                       "category_candidates": category_candidates,
                       "product_ids_by_brand_slug": product_ids_by_brand_slug,
                       "product_ids_by_category_slug": product_ids_by_category_slug}
    return _catalog_cache


def _phrase_pattern(phrase: str) -> re.Pattern | None:
    words = []
    for w in phrase.split():
        if not w:
            continue
        if w.startswith("ال") and len(w) > 2:
            # Arabic definite article: canonical category/brand names
            # commonly carry it ("الشوكولاتة") but a real free-text query
            # just as commonly drops it ("شوكولاتة تحت 50") -- found via a
            # real query that failed to resolve a category for exactly this
            # reason. Make a leading "ال" on any word optional rather than
            # required, in either direction.
            words.append("(?:ال)?" + re.escape(w[2:]))
        else:
            words.append(re.escape(w))
    if not words:
        return None
    return re.compile(r"\b" + r"\s+".join(words) + r"\b", re.IGNORECASE)


def _extract_first_phrase_match(
    text: str, candidates: list[tuple[str, str]]
) -> tuple[str | None, str, int]:
    """candidates: [(slug, phrase), ...]. Grouped into tiers by word count
    (longest tier first) so a multi-word name (e.g. 'Meat & Poultry') is
    preferred over a shorter one ('Meat') when both could match -- a
    longer tier match always wins outright, regardless of where in the
    text it falls. Returns (matched_slug_or_None,
    text_with_the_matched_phrase_removed, matched_word_count) -- removal
    matters: leaving a recognized category/brand name in the leftover
    free-text would make it ALSO a required BM25 text-match term, silently
    causing false zero-results (e.g. 'meat under 100' would then require
    the literal word 'meat' in a matching product's title, on top of the
    category filter). The word count lets the caller compare this match's
    specificity against a competing brand match over the same text.

    Sprint 5 Part 3: within a tier, if MULTIPLE different candidates all
    match (different words, same word-count -- e.g. both 'لبن' -> milk and
    'الشوفان' -> oats are real, valid single-word exact category names
    present in 'لبن الشوفان'), the tie is broken by word ORDER, not
    candidate-list order (an incidental DB row order that reflects
    nothing about the query itself -- confirmed via a real query where it
    picked the wrong side of a real compound concept purely by chance).

    Which direction depends on the query's language, because Arabic and
    English compound nouns put their head noun in OPPOSITE positions:
    Arabic idafa ('لبن الشوفان' = 'milk of oats') and adjective-follows-
    noun constructions put the head FIRST, so Arabic text prefers the
    LEFTMOST match. English noun-noun compounds are right-headed ('cream
    cheese' is a CHEESE, 'chocolate milk' is a MILK, 'milk chocolate' is a
    CHOCOLATE -- the second word names the category, the first modifies
    it), so non-Arabic text prefers the RIGHTMOST match. This is evidence-
    backed, not assumed: a first version always preferred leftmost, which
    correctly fixed the Arabic oat-milk case but was PROVEN wrong for
    English by a real, common product -- 'cream cheese' resolved to
    category=cream_11677 (leftmost 'cream') with a hard-filter leftover of
    'cheese', returning ZERO results, even though the real products (e.g.
    'Kiri Cream Cheese') are catalogued under 'cheese'; the awkward
    reversed phrasing 'cheese cream' accidentally worked by the same
    leftmost rule, which is the wrong way around. Reuses the existing
    is_arabic() helper (already used by _normalize_for_fuzzy) rather than
    adding new language detection. This only ever activates on a true
    same-length tie -- a single matching candidate in a tier is returned
    exactly as before, in either language."""
    prefer_leftmost = is_arabic(text)
    ordered = sorted(candidates, key=lambda pair: -len(pair[1].split()))
    idx = 0
    n = len(ordered)
    while idx < n:
        tier_len = len(ordered[idx][1].split())
        tier_end = idx
        while tier_end < n and len(ordered[tier_end][1].split()) == tier_len:
            tier_end += 1
        best: tuple[int, str, int] | None = None  # start, slug, phrase_word_count
        best_match = None
        for order_index in range(idx, tier_end):
            slug, phrase = ordered[order_index]
            pattern = _phrase_pattern(phrase)
            if pattern is None:
                continue
            m = pattern.search(text)
            if m is None:
                continue
            better = (
                best is None
                or (prefer_leftmost and m.start() < best[0])
                or (not prefer_leftmost and m.start() > best[0])
            )
            if better:
                best = (m.start(), slug, len(phrase.split()))
                best_match = m
        if best is not None:
            _, slug, word_count = best
            return slug, text[:best_match.start()] + text[best_match.end():], word_count
        idx = tier_end
    return None, text, 0


def _strip_leading_al_per_word(text: str) -> str:
    """Same rule _phrase_pattern already applies for exact matching --
    strip a leading Arabic definite article "ال" from each word -- reused
    here so fuzzy comparison gets the same tolerance: without it, a typo'd
    query missing the category name's "ال" (real query 'بسكوت' vs real
    category name 'البسكويت') pays both the typo AND the missing-article
    penalty at once and can fall below threshold that a typo alone
    wouldn't (76.9 vs 90.9 -- confirmed via direct testing)."""
    return " ".join(w[2:] if w.startswith("ال") and len(w) > 2 else w for w in text.split())


def _fuzzy_phrase_match(
    text: str, candidates: list[tuple[str, list[str]]], threshold: float,
    blocklist_norm: set[str], max_phrase_words: int,
    window_transforms: tuple = (lambda w: w,),
    word_exclusions_norm: frozenset[str] = frozenset(),
    protected_words_norm: frozenset[str] = frozenset(),
    protected_threshold: float = 95.0,
    candidate_product_ids: dict[str, set[int]] | None = None,
    corpus_protection: bool = True,
    strict_unattested: bool = False,
) -> tuple[str | None, str, int]:
    """Shared typo/near-miss tolerant matcher (rapidfuzz edit-distance
    ratio) used by both brand and category detection -- same algorithm,
    different candidate lists/thresholds/blocklists. candidates:
    [(slug, [norm_form_0, norm_form_1, ...]), ...], one normalized form
    per entry in window_transforms -- transform[i] applied to the window
    is scored against norm_forms[i], and the BEST score across all pairs
    wins (not just one fixed transform): category needs this because
    always stripping the Arabic definite article "ال" helps a query
    that's missing it entirely ('بسكوت' vs 'البسكويت') but HURTS a query
    that already has it, since stripping shortens an already-short word
    and makes an ordinary one-character typo in it fail threshold that it
    wouldn't have otherwise -- confirmed via a full regression sweep
    (research.md), not assumed. Scans every contiguous query word window
    (up to max_phrase_words); a blocklisted window (its default,
    untransformed form) never matches, regardless of score. Only the
    default (identity) form is used to reconstruct the returned remaining
    text, so leftover free text elsewhere in the query is never altered
    by a transform.

    Ties prefer the LONGER window, not whichever is scored first: found
    via a real bug -- a real 2-word brand ('Aqua Delta', 'Big Babol',
    'Dabur Amla', 'Fresh Days'...) was getting chopped down to a
    DIFFERENT, shorter, also-real brand ('Aqua', 'Big', 'Dabur', 'Fresh')
    because both scored a tied 100.0 and the shorter window is generated
    first. Same principle _extract_first_phrase_match already applies for
    exact matching (sorts candidates longest-phrase-first) -- fuzzy
    matching needs the same preference, just enforced as a tie-break
    here since scores (not just presence/absence of a match) are involved.

    word_exclusions_norm (Sprint 5 Part 2): unlike blocklist_norm, which
    only blocks a window whose FULL normalized text exactly equals a
    blocklisted phrase, this disqualifies any window that CONTAINS one of
    these words at all, regardless of window length or what else is in
    it. Deliberately kept separate and used sparingly (only for
    functional/grammatical words like "بدون" that could never
    legitimately be part of a real multi-word brand name, or for a
    caller-computed, per-call set of words with stronger exact-match
    evidence elsewhere) -- applying it broadly would break real brands
    that legitimately contain an ordinary product word ("My Meat", "Ahmed
    Tea", "Coffee Break", "Mc Sauce" -- see parse_keyword_query's own
    docstring), which must still be free to win fairly via the normal
    length-based competition.

    protected_words_norm (Sprint 5 Part 2): softer than
    word_exclusions_norm -- a window containing one of these words is not
    disqualified outright, only held to protected_threshold instead of
    the normal (lower) threshold. Used for words that ALREADY have their
    own real, authoritative catalog evidence (an exact category-name
    match on the leftover text -- see _protected_category_words) that a
    merely-plausible fuzzy brand coincidence shouldn't override, while
    still letting a genuinely near-exact/exact real brand name win (a
    full regression sweep found this distinction matters: "milk" scores
    only 88.9 against the coincidentally-similar real brand "Milka" --
    correctly blocked -- but "Hero" is ITSELF a real, exact brand name
    (scores 100) that also happens to collide with an unrelated real
    category also called "Hero" -- a blanket exclusion wrongly blocked
    this genuine brand too; the score gap between a coincidental
    resemblance and an actual exact/near-exact name is exactly what
    separates the two cases).

    Sprint 6: ALWAYS (no caller wiring needed) also raises a window to
    protected_threshold when the window's own words have strong REAL
    PRODUCT-CORPUS co-occurrence support (_corpus_cooccurrence_support >=
    _MIN_PRODUCT_COOCCURRENCE_SUPPORT) -- catalog-grounded evidence that
    the window is itself genuine, ordinary product language, not just a
    coincidentally-similar string. Found via real, fresh-session queries:
    "paste" (69 real products) scores exactly 80.0 against the unrelated
    category "Pasta"; "peas" (46 real products) scores 88.9 against the
    unrelated real brand "Pears"; "frozen peas" (12 real products contain
    both words) scores 81.8 against the unrelated category "Frozen Meat"
    purely because both phrases share the word "frozen". This is the
    exact same principle as protected_words_norm above, just with a
    different (corpus-frequency, not category-name-exactness) source of
    evidence.

    This protection is then PER-CANDIDATE exempted (via
    candidate_product_ids + _corpus_overlap_ratio) when the window's own
    real-product evidence substantially OVERLAPS that specific
    candidate's real product membership -- i.e. the window isn't just
    "a real word", it's specifically real evidence FOR this candidate,
    meaning it's a genuine spelling/grammar variant of the SAME entity,
    not a coincidence. This distinction could NOT be made by score alone:
    "جهينة" (a real spelling variant of the real brand alias "جهينه") and
    "detergent" (singular of the real category "Detergents") score 80.0
    and 94.7 respectively against their TRUE targets -- scores that
    overlap the SAME range as the confirmed coincidences above, since
    edit-distance ratio can't tell "same word, different spelling" apart
    from "different word, coincidentally similar spelling" (paste/pasta
    and جهينة/جهينه are both single-character-different, same length).
    89-90% (جهينة) down to 43% (كامل+الدسم vs its own category) of these
    genuine variants' real supporting products are ALSO real instances of
    the matched candidate, while all 3 confirmed coincidences above are
    at exactly 0% -- a large, clean, measured gap.

    A genuinely near-exact/exact real brand or category name (e.g.
    "Extra" typed alone, or a real typo like "milqa") is unaffected by
    either mechanism, since typos have ~0 corpus support and real short
    brand names are still free to win at protected_threshold or above.

    strict_unattested (Sprint 6 out-of-catalog grounding follow-up):
    additionally escalates a window with ZERO corpus occurrence of its OWN
    (not just well-attested-but-mismatched) to _UNATTESTED_WINDOW_MIN_SCORE.
    Deliberately opt-in, used ONLY by the role-aware conversational
    resolvers (resolve_category_hint/resolve_brand_text) -- NOT by
    parse_keyword_query's free-sentence scanning (_find_fuzzy_category/
    _find_fuzzy_brand's own default calls). Measured directly why this must
    stay scoped: parse_keyword_query's full-catalog typo sweep
    (tests/integration/test_category_fuzzy.py) generates thousands of
    single-character-substitution typos on real, often short, brand/
    category names -- the overwhelming majority have ZERO literal corpus
    occurrence of their OWN misspelled form (a random substitution rarely
    happens to spell another real corpus word), so "unattested" is the
    NORMAL shape of an ordinary short-word typo in that context, not a red
    flag; turning this on there measured cat_typo/brand_typo recall
    collapsing (0.78->0.74, 0.77->0.64). The conversational resolvers see a
    different, much narrower input instead -- a short, deliberate LLM-
    extracted phrase naming ONE concept -- where the real bug this closes
    (category_hint "mobile phone" hijacking the unrelated category
    "Honey") actually lives; see _UNATTESTED_WINDOW_MIN_SCORE."""
    words = text.split()
    if not words:
        return None, text, 0

    best: tuple[float, int, str, int, int] | None = None  # score, length, slug, start, end
    for start in range(len(words)):
        max_len = min(max_phrase_words, len(words) - start)
        for length in range(1, max_len + 1):
            end = start + length
            if any(_normalize_for_fuzzy(w) in word_exclusions_norm for w in words[start:end]):
                continue
            window = _normalize_for_fuzzy(" ".join(words[start:end]))
            if window in blocklist_norm:
                continue
            base_effective_threshold = threshold
            if protected_words_norm and any(_normalize_for_fuzzy(w) in protected_words_norm for w in words[start:end]):
                base_effective_threshold = max(threshold, protected_threshold)
            # corpus_protection=False is used by the role-aware resolvers
            # (resolve_brand_text / resolve_category_hint): all of this
            # machinery exists to guess whether an arbitrary word was even
            # MEANT as a brand/category, and when the understanding layer
            # has already said so explicitly there is nothing left to guess.
            window_ids = _corpus_cooccurrence_ids(words[start:end]) if corpus_protection else set()
            window_is_well_attested = len(window_ids) >= _MIN_PRODUCT_COOCCURRENCE_SUPPORT
            # Sprint 6 out-of-catalog grounding follow-up: a window with ZERO
            # real corpus occurrence of its own is held to
            # _UNATTESTED_WINDOW_MIN_SCORE instead of the plain (lower)
            # threshold, which previously gave it LESS scrutiny than a
            # well-attested-but-mismatched window. Real bug this closes:
            # category_hint "mobile phone" -> the single-word sub-window
            # "phone" scores exactly 80.0 against the unrelated real
            # category "Honey", and "phone" occurs in ZERO real products --
            # see _UNATTESTED_WINDOW_MIN_SCORE for why this is a score
            # threshold, not an edit-distance one.
            window_is_unattested = strict_unattested and corpus_protection and not window_ids
            scored_windows = [transform(window) for transform in window_transforms]
            for slug, norm_forms in candidates:
                effective_threshold = base_effective_threshold
                if window_is_well_attested:
                    candidate_ids = candidate_product_ids.get(slug) if candidate_product_ids else None
                    if _corpus_overlap_ratio(window_ids, candidate_ids) < _MIN_CANDIDATE_OVERLAP_RATIO:
                        effective_threshold = max(effective_threshold, protected_threshold)
                elif window_is_unattested:
                    effective_threshold = max(effective_threshold, _UNATTESTED_WINDOW_MIN_SCORE)
                score = max(fuzz.ratio(w, n) for w, n in zip(scored_windows, norm_forms))
                if score >= effective_threshold and (best is None or (score, length) > (best[0], best[1])):
                    best = (score, length, slug, start, end)

    if best is None:
        return None, text, 0
    _, length, slug, start, end = best
    return slug, " ".join(words[:start] + words[end:]), length


_BRAND_FUZZY_THRESHOLD = 80  # research.md: lowest score seen on real
# misspelled-brand test cases ("milqa" vs "Milka" = 80.0). Some coincidental
# collisions with short, unrelated real brand names that merely resemble a
# generic word score just as high or higher and aren't distinguishable by
# edit-distance ratio alone -- not fixable by threshold, so the specific
# words confirmed to do this are excluded outright below instead.
_BRAND_MAX_PHRASE_WORDS = 3  # real brand names in this catalog are short

# Generic product/category-adjacent words confirmed via direct testing to
# coincidentally score >= _BRAND_FUZZY_THRESHOLD against an unrelated real
# catalog brand purely by edit-distance chance -- e.g. "wafers" scores 83.3
# against the real brand "Wafeer", higher than the genuine typo "milqa" vs
# "Milka" (80.0) scores against its real target, so no threshold can keep
# one and drop the other. Never treated as a brand match regardless of
# score. Deliberately a short, explicit list (not a general dictionary) --
# add to it only when a new false positive is actually found, not guessed.
_BRAND_FUZZY_BLOCKLIST = {
    "wafer", "wafers", "fish", "chip", "chips", "salt", "flour", "yeast",
    "baking", "wine", "dairy", "clean", "spicy",
    "مايونيز", "حلوى", "خميرة", "زبادي",
    # Found via a real live query: "لين" (typo of "لبن"/milk) scored 85.7
    # against the real brand "لينو" -- higher than most genuine typos --
    # hard-filtering to Lino-only products and excluding milk entirely.
    "لين", "شامبو",
    # Found via real adversarial test sentences (not stopwords -- ordinary
    # descriptive/substantive words, so not part of the stopword sweep
    # below): "breast" (from "chiken breast") vs real brand "Beast"
    # (90.9), "براند" (Arabic for "brand", used generically/meta) vs real
    # brand alias "براون"/Braun (80.0).
    "breast", "براند",
    # Found via a systematic sweep of ~144 common EN/AR filler/stopwords
    # through the real resolver, not guessed one at a time (research.md):
    # 31 of them scored >= threshold against a real, unrelated brand --
    # e.g. "or" vs the real brand "Ors" (80.0), "sure"/"fine"/"today"/
    # "one" ARE literally real brand names (100.0). This is a large,
    # bounded, confirmed list, not an open-ended one -- the blocklist
    # mechanism itself doesn't change, it's just longer.
    "an", "or", "for", "any", "sure", "fine", "today", "in", "on", "at",
    "much", "but", "if", "give", "shall", "down", "off", "over", "again",
    "one",
    "في", "إلى", "الى", "هل", "لا", "كام", "لأ", "انا", "انتي", "فيه",
    "فين",
    # Found on the SECOND sweep pass, after blocklisting "اي" for category
    # (vs "الشاي"/tea): it fell through to a DIFFERENT real brand, "كاى"
    # (Kayy, 80.0) -- blocklisting one match type doesn't block others for
    # the same word, so both needed fixing.
    "اي",
    # Found via a real live query, same class as "wafers"/"Wafeer" above:
    # "cookie" vs the real, distinct brand "Cooker" (83.3) hijacked the
    # entire query, returning Cooker-brand canned/pantry goods and zero
    # real cookies; "dish" vs the real, distinct brand "Modish" (80.0)
    # hijacked to Modish-brand hair-care products. Neither "cookie" nor
    # "dish" is itself a real raw brand alias or canonical brand
    # (confirmed against the live table) -- ordinary product words, not
    # brand mentions. "Cooker"/"Modish" themselves are untouched: an
    # explicit search for either real brand still resolves it.
    "cookie", "dish",
    # Found via Sprint 5 Part 3 fat-modifier investigation: "fat" (from
    # "low fat"/"full fat") scores 85.7 against the real, unrelated brand
    # "Fast" -- higher than most genuine typos. The milk-specific "full
    # fat milk"/"low fat milk" phrasings are consumed whole by
    # _CATEGORY_COMPOUND_ALIASES before brand matching ever sees a bare
    # "fat", but the word can still end up alone as leftover text in other
    # phrasings (e.g. a fat-modifier on a non-milk product), so it's
    # blocklisted here too, same as "wafers"/"cookie" above.
    "fat",
}
_BRAND_FUZZY_BLOCKLIST_NORM = {_normalize_for_fuzzy(w) for w in _BRAND_FUZZY_BLOCKLIST}

# Functional/grammatical words that must never be part of ANY brand-fuzzy
# window, regardless of window length or what else the window contains --
# see _fuzzy_phrase_match's word_exclusions_norm docstring for why this is
# deliberately separate from (and much smaller than) _BRAND_FUZZY_BLOCKLIST
# above. Found via a real live query (Sprint 5 Part 2 research spike):
# "لبن بدون" (2-word window, "milk without") scores exactly 80.0 against
# the real brand alias "بون بون" (Bonbons) -- purely coincidental, and not
# catchable by blocklisting "بدون" alone (blocklist_norm only blocks a
# window whose FULL text matches, not a window merely containing the
# word), since "بدون" alone only scores 54.5 against Bonbons (below
# threshold) -- the collision is specific to the 2-word combination.
# "بدون" ("without") is a pure negation particle -- unlike "meat"/"tea"/
# "sauce", it could never legitimately be a word inside a real product
# brand name, so excluding it from every window it touches carries no
# risk to real multi-word brands (contrast with parse_keyword_query's own
# "My Meat"/"Ahmed Tea" examples, which must still compete fairly).
_BRAND_FUZZY_WORD_EXCLUSIONS = {"بدون"}
_BRAND_FUZZY_WORD_EXCLUSIONS_NORM = frozenset(_normalize_for_fuzzy(w) for w in _BRAND_FUZZY_WORD_EXCLUSIONS)


def _find_fuzzy_brand(
    text: str, brands: list[dict], protected_category_words_norm: frozenset[str] = frozenset(),
) -> tuple[str | None, str, int]:
    candidates = [(b["slug"], [b["norm"]]) for b in brands]
    return _fuzzy_phrase_match(
        text, candidates, _BRAND_FUZZY_THRESHOLD, _BRAND_FUZZY_BLOCKLIST_NORM, _BRAND_MAX_PHRASE_WORDS,
        word_exclusions_norm=_BRAND_FUZZY_WORD_EXCLUSIONS_NORM,
        protected_words_norm=protected_category_words_norm,
        candidate_product_ids=_get_catalog_names()["product_ids_by_brand_slug"],
    )


_CATEGORY_FUZZY_THRESHOLD = 80  # same reasoning/value as brand (research.md)
_CATEGORY_MAX_PHRASE_WORDS = 6  # real max word count among canonical category names

# Same rationale as _BRAND_FUZZY_BLOCKLIST -- confirmed via direct testing
# (not guessed) that these coincidentally score >= _CATEGORY_FUZZY_THRESHOLD
# against an unrelated real category name. "price" vs the real "Rice"
# category scores 88.9 -- higher than most genuine category typos -- and
# broke an existing regression test the moment fuzzy fallback was added.
#
# The second group below is a different, systemic conflict found by
# sweeping every real canonical brand name against every real category
# name: category and brand are resolved as COMPETING matches over the
# same text (parse_keyword_query), with the longer/more specific match
# normally winning -- but when a brand
# and a category tie at the exact same word length (e.g. real brand
# "Milka" vs the real "Milk" category, both single-word, both scoring
# 88.9/100), length alone can't break the tie. These are the ones found
# where letting category win the tie was wrong -- a bare query "milka"
# resolved to category='milk_29' instead of brand='milka' until they were
# excluded here, so the length comparison correctly falls through to
# brand instead. Exact same-string ties (e.g. brand "Corona" / category
# "Corona") aren't included: those are resolved by exact category
# matching, a pre-existing, deliberate precedence this doesn't alter.
_CATEGORY_FUZZY_BLOCKLIST = {
    "price",
    "venus", "cheesa", "break", "dream", "farm cheese", "delia", "milka",
    "astra", "beast", "chocodate", "nucream",
    # Found via the same systematic filler/stopword sweep as the brand
    # blocklist above: "me" vs the real category "Men" (80.0), and three
    # Arabic prepositions/pronouns that only cross threshold once the
    # "ال"-stripping tolerance is applied (see _fuzzy_phrase_match) --
    # "من" (from) vs "السمن" (ghee, 80.0), "زي" (like) vs "الزيت" (oil,
    # 80.0), "اي" (any/which) vs "الشاي" (tea, 80.0).
    "me", "من", "زي", "اي",
}
_CATEGORY_FUZZY_BLOCKLIST_NORM = {_normalize_for_fuzzy(w) for w in _CATEGORY_FUZZY_BLOCKLIST}


def _find_fuzzy_category(
    text: str, category_candidates: list[tuple[str, str, str]],
    blocklist_norm: frozenset[str] = _CATEGORY_FUZZY_BLOCKLIST_NORM,
    strict_unattested: bool = False,
) -> tuple[str | None, str, int]:
    """Fallback for when the exact/definite-article-tolerant match
    (_extract_first_phrase_match) finds nothing -- catches a genuine
    misspelling of a category name ('بسكوت' for 'البسكويت') that literal
    matching can't. Tries both the normal comparison AND an "ال"-stripped
    one (mirrors _phrase_pattern's exact-match tolerance), keeping
    whichever scores higher -- see _fuzzy_phrase_match for why always
    stripping regressed the rest of the sweep.

    blocklist_norm defaults to the legacy query-specific
    _CATEGORY_FUZZY_BLOCKLIST (needed by parse_keyword_query/System A,
    which fuzzy-matches arbitrary FULL SENTENCES where an ordinary word
    can coincidentally resemble a category). resolve_category_hint (the
    conversational path) passes an empty set instead: its input is
    already the LLM's short, deliberate category description, not an
    arbitrary sentence, so the blocklist's query-specific entries (built
    for full-sentence false positives) don't apply -- the generic,
    catalog-evidence corpus-overlap check is what protects that path."""
    candidates = [
        (slug, [norm, _strip_leading_al_per_word(norm)])
        for slug, _name, norm in category_candidates
    ]
    return _fuzzy_phrase_match(text, candidates, _CATEGORY_FUZZY_THRESHOLD,
                                blocklist_norm, _CATEGORY_MAX_PHRASE_WORDS,
                                window_transforms=(lambda w: w, _strip_leading_al_per_word),
                                candidate_product_ids=_get_catalog_names()["product_ids_by_category_slug"],
                                strict_unattested=strict_unattested)


_CATEGORY_SINGULAR_ALIAS_SLUGS = set(_CATEGORY_SINGULAR_ALIASES.keys())
_CATEGORY_SINGULAR_ALIAS_PHRASES = {
    (slug, alias) for slug, aliases in _CATEGORY_SINGULAR_ALIASES.items() for alias in aliases
}


def _resolve_category(
    text: str, category_candidates: list[tuple[str, str, str]]
) -> tuple[str | None, str, int]:
    """Exact/"ال"-tolerant match first, fuzzy fallback second -- same
    two-step category resolution used throughout, just packaged as one
    call so parse_keyword_query can compare its result against brand's.

    If the exact match came from a singular-form alias (e.g. "لحمة" for
    'meat'), also check whether fuzzy matching finds a LONGER, more
    specific compound match on the same text (e.g. "لحمة معلبة" -> canned
    meat, not just plain "meat") and prefer that -- fuzzy-matched against
    the REAL category names only, with the alias phrases themselves
    excluded from that check. Found as two real, layered regressions
    while adding the aliases: (1) "لحمة" is a literal 1-word exact match,
    so it always won immediately and never let "لحمة معلبة" reach fuzzy
    matching at all; (2) even after routing it through fuzzy matching too,
    the alias "لحمة" scores a perfect 100 against ITSELF as a 1-word
    window, which beats the real, longer "لحوم معلبة" match (90.0) on raw
    score before length ever gets compared -- excluding the alias phrases
    from this specific fuzzy pass removes that self-competition. Scoped to
    only the alias-bearing categories, not every query, so this doesn't
    cost every resolution the extra fuzzy pass."""
    plain_candidates = [(slug, name) for slug, name, _norm in category_candidates]
    slug, remaining, length = _extract_first_phrase_match(text, plain_candidates)
    if slug is None:
        return _find_fuzzy_category(text, category_candidates)
    if slug in _CATEGORY_SINGULAR_ALIAS_SLUGS:
        real_candidates = [
            c for c in category_candidates if (c[0], c[1]) not in _CATEGORY_SINGULAR_ALIAS_PHRASES
        ]
        fuzzy_slug, fuzzy_remaining, fuzzy_length = _find_fuzzy_category(text, real_candidates)
        if fuzzy_slug is not None and fuzzy_length > length:
            return fuzzy_slug, fuzzy_remaining, fuzzy_length
    return slug, remaining, length


# Arabic/English FREE_FROM(X) operator MARKERS -- "من غير سكر", "بدون
# سكر", "without sugar", "no sugar", "free from gluten" all express the
# same semantic relation, just with different wording that (unlike "X
# free"/"خالي من X", which already IS the literal stored category name
# and needs no special handling at all) never textually matches any real
# category name on its own. Only the MARKER is matched here (not a
# captured word) -- see _resolve_negation_free_from for why.
_NEGATION_PATTERNS = [
    re.compile(r"\bمن\s+غير\b"),
    re.compile(r"\bبدون\b"),
    re.compile(r"\bwithout\b", re.IGNORECASE),
    re.compile(r"\bno\b", re.IGNORECASE),
    re.compile(r"\bfree\s+from\b", re.IGNORECASE),
]

# Sprint 5 Part 3 adversarial audit: a real modifier word can sit between
# the negation marker and the actual negated noun ("no ADDED sugar",
# "without ARTIFICIAL sugar", "بدون سكر مضاف" -- Arabic post-nominal
# adjectives already worked by accident since the noun directly follows
# the marker there, but English pre-nominal modifiers broke a version
# that only ever looked at the single word immediately after the marker:
# it captured "added"/"artificial", _find_free_from_category correctly
# found nothing for those, and the negation was silently dropped --
# "no added sugar" then fell through to ordinary resolution, where
# "sugar" matched the literal POSITIVE "Sugar" category, the exact
# invariant this whole mechanism exists to prevent. This bounds how many
# words past the marker are checked, not what they can be -- still no
# concept-by-concept mapping.
_FREE_FROM_WINDOW_WORDS = 3

# Words that mark a category name as expressing "free from <something>" --
# every real free-from category in this catalog uses one of these
# (confirmed via a full scan of every category name containing "free" or
# "خالي": Gluten Free/خالي من الجلوتين, Lactose Free/خالي من اللاكتوز,
# Sugar Free/خالي من السكر -- exactly 3, all following this same
# convention, none are exceptions).
_FREE_FROM_MARKERS_NORM = frozenset(_normalize_for_fuzzy(w) for w in ("free", "خالي", "خال"))


def _find_free_from_category(
    concept_word: str, category_candidates: list[tuple[str, str, str]]
) -> str | None:
    """Sprint 5 Part 3: replaces Part 2's hardcoded _FREE_FROM_COUNTERPART
    (a POSITIVE-category-slug -> free-from-category-slug dict, one entry
    per concept, that could only ever redirect a concept that ALSO had its
    own plain positive category -- e.g. it could redirect "سكر"/sugar
    since sugar_5711 exists, but had no way to redirect "لاكتوز"/lactose
    at all, since there is no plain "Lactose" category to look up a
    counterpart FROM in the first place, even though the real
    lactose-free category obviously exists).

    This is catalog-grounded instead: it doesn't need the concept to have
    its own category -- it directly searches every real category name for
    one that contains BOTH a free-from marker word (_FREE_FROM_MARKERS_NORM)
    AND a word matching the concept (exact, or "ال"-tolerant, the same
    tolerance _phrase_pattern already gives ordinary category matching).
    Only 3 such categories exist in the whole catalog (a small, cheap
    search), so this generalizes safely to any concept a free-from
    category actually exists for, without inventing one that doesn't --
    e.g. "milk without lactose" still correctly finds nothing extra beyond
    what already existed (lactose_free_32297), while "بدون جلوتين" now
    also correctly resolves (gluten_free_27956), which the old hardcoded
    dict never covered because gluten was never added to it."""
    concept_norm = _normalize_for_fuzzy(concept_word)
    concept_destripped = _strip_leading_al_per_word(concept_norm)
    best: tuple[float, str] | None = None
    for slug, _name, norm in category_candidates:
        words = norm.split()
        if not any(w in _FREE_FROM_MARKERS_NORM for w in words):
            continue
        for w in words:
            if w in _FREE_FROM_MARKERS_NORM:
                continue
            w_destripped = _strip_leading_al_per_word(w)
            score = max(
                fuzz.ratio(concept_norm, w), fuzz.ratio(concept_norm, w_destripped),
                fuzz.ratio(concept_destripped, w), fuzz.ratio(concept_destripped, w_destripped),
            )
            if score >= _CATEGORY_FUZZY_THRESHOLD and (best is None or score > best[0]):
                best = (score, slug)
    return best[1] if best else None


def _resolve_negation_free_from(
    text: str, category_candidates: list[tuple[str, str, str]]
) -> tuple[str, str] | None:
    """If `text` contains a negation marker followed, within
    _FREE_FROM_WINDOW_WORDS words, by a word X that has a real,
    authoritative FREE_FROM(X) category in the catalog (see
    _find_free_from_category), return (free_from_slug,
    text_with_the_marker_THROUGH_the_matched_word_removed) -- the
    negation wording itself ("من غير"/"بدون"/"without"/"no"/"free from"),
    and any modifier word in between ("added"/"artificial"/"مضاف"), are
    removed too, so neither leaks into query_text. Tries each word in the
    window in order and stops at the first one with real catalog
    evidence, so "no added sugar" finds nothing for "added" and then
    correctly matches "sugar"; "بدون سكر مضاف" matches "سكر" immediately
    (Arabic post-nominal adjectives never needed the window at all).
    Returns None if no word in the window has a real free-from category,
    leaving the normal competing-match resolution untouched and the full
    phrase preserved as free text downstream -- e.g. "milk without
    lactose syrup" (not a real concept with a free-from category)
    correctly does nothing here."""
    for pattern in _NEGATION_PATTERNS:
        m = pattern.search(text)
        if not m:
            continue
        for wm in list(re.finditer(r"\S+", text[m.end():]))[:_FREE_FROM_WINDOW_WORDS]:
            free_from_slug = _find_free_from_category(wm.group(), category_candidates)
            if free_from_slug is not None:
                return free_from_slug, text[:m.start()] + text[m.end() + wm.end():]
    return None


def _protected_category_words(text: str, category_candidates: list[tuple[str, str, str]]) -> frozenset[str]:
    """Words in `text` that are THEMSELVES an exact (non-fuzzy) category-
    name/alias match -- real, authoritative catalog evidence that a
    weaker FUZZY brand coincidence must not steal (Sprint 5 Part 2). Used
    only for the brand RE-RESOLUTION pass on a category's leftover text,
    never the first competing-match pass, so a real multi-word brand that
    legitimately contains a product word ("My Meat", "Ahmed Tea") still
    competes fairly there via the existing length-based rule -- this only
    guards the narrower case where category has ALREADY won a separate,
    stronger span (e.g. "lactose free"), leaving a single ordinary
    product word ("milk") sitting alone in the leftover, where a fuzzy
    brand match (e.g. "Milka") would otherwise face no real competition
    and incorrectly claim it."""
    plain_candidates = [(slug, name) for slug, name, _norm in category_candidates]
    protected: set[str] = set()
    for word in text.split():
        slug, _, _ = _extract_first_phrase_match(word, plain_candidates)
        if slug is not None:
            protected.add(_normalize_for_fuzzy(word))
    return frozenset(protected)


_PRODUCT_PHRASE_MAX_WINDOW = 4  # real multi-word product descriptors in
# this catalog ("extra virgin olive oil") are short; matches _BRAND_MAX_
# PHRASE_WORDS + 1 headroom, not picked independently.


def _longest_evidenced_phrase_window(text: str) -> tuple[int, int] | None:
    """Sprint 6: the LONGEST contiguous word span (>= 2 words) in `text`
    whose words all co-occur in at least _MIN_PRODUCT_COOCCURRENCE_SUPPORT
    real products (see _corpus_cooccurrence_support) -- i.e. a genuine,
    catalog-attested multi-word product descriptor phrase, as opposed to
    one word of it happening to also be a real brand/category name.
    Returns (start, end) word indices, or None if no such span exists.
    Only ever consulted by _resolve_leftover_brand below, to decide
    whether a brand match found on LEFTOVER text is fragmenting a real,
    longer, more-specific product phrase -- never during the first-pass
    category-vs-brand competition, where a real multi-word brand that
    legitimately contains an ordinary product word ("Ahmed Tea", "Coffee
    Break") must still be free to win fairly on its own length."""
    words = text.split()
    best: tuple[int, int] | None = None
    for start in range(len(words)):
        max_len = min(_PRODUCT_PHRASE_MAX_WINDOW, len(words) - start)
        for length in range(2, max_len + 1):
            end = start + length
            if _corpus_cooccurrence_support(words[start:end]) >= _MIN_PRODUCT_COOCCURRENCE_SUPPORT:
                if best is None or length > (best[1] - best[0]):
                    best = (start, end)
            else:
                break  # extending further from `start` can only add words; if THIS length
                # already lacks support, a longer one starting at the same position won't
                # gain it back (co-occurrence support is monotonically non-increasing as the
                # required word set grows), so stop extending this `start` early.
    return best


def _resolve_leftover_brand(
    remaining: str, brands: list[dict], category_candidates: list[tuple[str, str, str]]
) -> tuple[str | None, str]:
    """Shared by both leftover-brand-resolution call sites in
    parse_keyword_query (the main category-wins-tie branch and the
    negation/free-from branch) -- previously each called _find_fuzzy_brand
    directly; Sprint 6 adds one more real-catalog-evidence check on top,
    so it's centralized here rather than duplicated twice.

    After the existing category-name-based protection (Sprint 5 Part 2),
    ALSO reject a brand match if the leftover contains a real, catalog-
    attested multi-word product phrase (_longest_evidenced_phrase_window)
    that is STRICTLY LONGER than the brand's own matched span and
    overlaps it -- e.g. "extra virgin olive" (after category already
    consumed "oil") has real product-corpus evidence as a 3-word (or at
    minimum 2-word) unit, longer than the 1-word brand match "extra" ->
    the real brand "Extra"; the longer, more specific real product
    phrase wins, the SAME 'more of the query covered wins' principle
    parse_keyword_query's own docstring already uses for the top-level
    category-vs-brand competition, just applied against real corpus
    evidence instead of another category/brand candidate. A real brand
    match that already covers the ENTIRE evidenced phrase (e.g. "Ahmed
    Tea" when the evidenced span is also exactly those same 2 words) is
    never rejected, since the evidenced span is then not LONGER than the
    match -- true multi-word brands remain unaffected."""
    protected = _protected_category_words(remaining, category_candidates)
    brand_slug, new_remaining, brand_match_len = _find_fuzzy_brand(
        remaining, brands, protected_category_words_norm=protected
    )
    if brand_slug is None:
        return None, new_remaining
    evidenced = _longest_evidenced_phrase_window(remaining)
    if evidenced is not None and (evidenced[1] - evidenced[0]) > brand_match_len:
        return None, remaining
    return brand_slug, new_remaining


def parse_keyword_query(text: str) -> FilterSet:
    """Category and brand are resolved as COMPETING matches over the same
    text, not strictly one-then-the-other: a real multi-word brand name
    can contain a real category word inside it ('Ahmed Tea', 'Coffee
    Break', 'Mc Sauce', 'My Meat') -- if category always ran first, it
    would grab just that one word out of the middle of the brand name
    (having no way to know a longer, more specific brand match exists for
    the same span), fragmenting the brand and leaving a meaningless
    remainder; if brand always ran first, the same problem hits in
    reverse for a real category word that resembles a real brand
    ('chocolate' vs the real brand 'Chocodate'). Whichever match covers
    MORE of the query wins and is removed first, then the other type is
    re-resolved on what's actually left -- confirmed via a real
    catalog-wide sweep (every real category and every real brand name, in
    3 forms each), not a guess for one name. A genuine tie (both match
    the same single word) falls back to category winning by default,
    EXCEPT the specific collisions in _CATEGORY_FUZZY_BLOCKLIST where
    that was confirmed wrong."""
    price_min, price_max, remaining = extract_price_constraints(text)
    catalog = _get_catalog_names()

    negation_result = _resolve_negation_free_from(remaining, catalog["category_candidates"])
    if negation_result is not None:
        category_slug, remaining = negation_result
        # Same leftover-brand-collision risk as the main branch below (e.g.
        # "شوكولاتة بدون سكر" leaves "شوكولاتة" alone in the leftover after
        # the free-from redirect, which a fuzzy brand match ("شوكودات"/
        # Chocodate) would otherwise steal) -- found during the Sprint 5
        # Part 3 audit, fixed the same way.
        brand_slug, remaining = _resolve_leftover_brand(remaining, catalog["brands"], catalog["category_candidates"])
        return FilterSet(
            category=category_slug,
            price_min=price_min,
            price_max=price_max,
            brand=brand_slug,
            query_text=remaining.strip() or None,
        )

    cat_slug, cat_remaining, cat_len = _resolve_category(remaining, catalog["category_candidates"])
    brand_slug, brand_remaining, brand_len = _find_fuzzy_brand(remaining, catalog["brands"])

    if brand_len > cat_len:
        remaining = brand_remaining
        category_slug, remaining, _ = _resolve_category(remaining, catalog["category_candidates"])
    else:
        category_slug = cat_slug
        remaining = cat_remaining
        brand_slug, remaining = _resolve_leftover_brand(remaining, catalog["brands"], catalog["category_candidates"])

    return FilterSet(
        category=category_slug,
        price_min=price_min,
        price_max=price_max,
        brand=brand_slug,
        query_text=remaining.strip() or None,
    )


# ---------------------------------------------------------------------
# Role-aware resolvers (Sprint 6 semantic-role refactor).
#
# Everything above this line resolves entity identity from an ARBITRARY
# sentence, where the role of each word is unknown -- that unknown is the
# reason the blocklists, word exclusions and corpus-evidence protections
# above exist at all, and that path now serves only the System A keyword
# baseline (/search/keyword, FR-014).
#
# The conversational path no longer asks those questions. The LLM has
# already said "this span is a brand" / "this span describes a category",
# so these two entry points take a short, single-role string and only
# answer the CATALOG question: does this correspond to a real row? No
# semantic blocklist, no word exclusion, no corpus-overlap protection --
# there is no competing role left to protect against.
# ---------------------------------------------------------------------

def resolve_brand_text(brand_text: str | None) -> str | None:
    """Canonical brand slug for a brand the user explicitly named, or None
    if nothing in the real brand table is close enough. Typo-tolerant on
    purpose ("Juhyna" -> juhayna): a misspelled brand is still
    unambiguously a BRAND mention, which is exactly the case fuzzy
    matching is good at once the role is settled.

    Corpus protection stays ON here too, even though the role is already
    known. Measured directly: it costs nothing on real typo cases (Extra,
    Juhyna, Nestlle, Pepsii, Colgat, جهينه all resolve identically with it
    on or off), and it is the only thing standing between "الوادي" (a
    common Arabic word for "valley/oasis", real on Wadi Dates products)
    and a false, confident match onto the unrelated real brand "El
    Bawadi" -- a coincidental fuzzy lookalike, exactly the failure shape
    corpus protection exists to catch, and knowing the ROLE is "brand"
    does not settle WHICH brand any more than knowing the role is
    "category" settles which category (see resolve_category_hint).

    Falls back to the SAME general corpus spelling-corrector
    ranking.py's free-text search path already relies on
    (correct_unmatched_terms) when the direct fuzzy attempt above finds
    nothing -- real gap this closes: a one-substitution brand typo
    ("pebsi") scores exactly _BRAND_FUZZY_THRESHOLD (80) against its real
    target ("pepsi"), but the unattested-window protection above (needed
    for "fruit"/"phone"-shaped false positives) raises the effective
    threshold to _UNATTESTED_WINDOW_MIN_SCORE (82) whenever the typo
    itself -- as any genuine typo will -- has zero corpus presence of its
    own, silently rejecting a correction the corpus already proves is
    safe. correct_unmatched_terms answers a DIFFERENT, more precise
    question (does the CANDIDATE correction have real, well-attested
    support -- SPELLING_CORRECTION_MIN_TARGET_DOCS=3), so a corrected
    word is re-checked through the exact same corpus-protected fuzzy
    match above, not trusted directly -- it can only ever resolve to a
    brand that already independently clears every existing safety net."""
    if not brand_text or not brand_text.strip():
        return None
    catalog = _get_catalog_names()
    candidates = [(b["slug"], [b["norm"]]) for b in catalog["brands"]]
    slug, _remaining, _length = _fuzzy_phrase_match(
        brand_text, candidates, _BRAND_FUZZY_THRESHOLD, set(), _BRAND_MAX_PHRASE_WORDS,
        corpus_protection=True, strict_unattested=True,
    )
    if slug is not None:
        return slug

    corrected = correct_unmatched_terms(get_index(), brand_text)
    if not corrected:
        return None
    slug, _remaining, _length = _fuzzy_phrase_match(
        corrected, candidates, _BRAND_FUZZY_THRESHOLD, set(), _BRAND_MAX_PHRASE_WORDS,
        corpus_protection=True, strict_unattested=True,
    )
    return slug


def resolve_category_hint(category_hint: str | None) -> str | None:
    """Canonical category slug for the LLM's semantic category hint, or
    None if it cannot be grounded. Exact/"ال"-tolerant phrase matching
    first (the hint is usually already close to a real category name),
    fuzzy second for wording/spelling differences. Returning None is a
    valid, expected outcome -- an ungrounded hint simply leaves the search
    to product_query/BM25 rather than inventing a canonical category."""
    if not category_hint or not category_hint.strip():
        return None
    catalog = _get_catalog_names()
    plain = [(slug, name) for slug, name, _norm in catalog["category_candidates"]]
    slug, _remaining, _length = _extract_first_phrase_match(category_hint, plain)
    if slug is not None:
        return slug
    # Corpus protection stays ON here, unlike resolve_brand_text. Knowing
    # the ROLE ("this describes a category") does not settle IDENTITY: the
    # hint is the user's own wording and frequently names something that is
    # not a real category at all ("tomato paste"), where nearest-neighbour
    # fuzzy matching would happily return the closest-looking real row
    # ("Pasta", 80.0) -- inventing a canonical category rather than
    # admitting it cannot be grounded. The corpus-overlap evidence check
    # separates a genuine wording/spelling difference for the SAME thing
    # ("detergent" -> Detergents, 63% of its real products overlap) from a
    # coincidental lookalike ("tomato paste" -> Pasta, 0% overlap), which
    # is precisely the question being asked here.
    # No query-specific blocklist here (Sprint 6 Finding C): category_hint
    # is already the LLM's deliberate category description, not an
    # arbitrary sentence -- the generic corpus-overlap check above is what
    # protects this path, not a hand-curated word list.
    slug, _remaining, _length = _find_fuzzy_category(
        category_hint, catalog["category_candidates"], blocklist_norm=frozenset(),
        strict_unattested=True,
    )
    return slug


def resolve_free_from(free_from: str | None) -> str | None:
    """Canonical slug of a real "X Free" category for an exclusion the
    user expressed, or None. Thin wrapper over _find_free_from_category so
    the conversational path never reaches into keyword_baseline internals
    -- and so an exclusion with no real free-from category in the catalog
    stays UNRESOLVED rather than collapsing into the positive concept."""
    if not free_from or not free_from.strip():
        return None
    return _find_free_from_category(free_from, _get_catalog_names()["category_candidates"])


def search_keyword(text: str, limit: int = 20, offset: int = 0) -> SearchResult:
    filters = parse_keyword_query(text)
    return SearchAdapter().search(filters, limit=limit, offset=offset)
