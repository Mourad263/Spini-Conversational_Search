"""Hand-rolled BM25 ranking over a Stage-1 (Postgres hard-filter) candidate
set (research.md §2, Stage 2). Corpus statistics come from the full-corpus
SearchIndex, never recomputed per-query.
"""

from __future__ import annotations

from rapidfuzz import fuzz, process

from src.search_engine.index import SearchIndex
from src.search_engine.tokenizer import tokenize

K1 = 1.4
B = 0.75
EXACT_SKU_BOOST = 100.0
URL_KEY_TOKEN_BOOST = 2.0

# Word-level spelling correction for query terms that have zero literal
# BM25 matches (research.md) -- same rapidfuzz/threshold philosophy as
# brand/category fuzzy matching, but the correction VOCABULARY is
# index.postings itself (real tokens from actual product names), since
# that's exactly what a re-run of rank() can act on -- a category-only
# word wouldn't help here even if it scored well.
#
# SPELLING_CORRECTION_THRESHOLD=80 alone is not enough on this vocabulary:
# it's 24,544 tokens (vs. ~1,691 brands/~488 categories), and includes
# thousands of short, rare fragments (numbers, abbreviations, one-off
# typos already baked into some product's own name). Confirmed directly:
# "milq" (typo of "milk") scores 85.7 against the real word "mil" -- a
# 3-letter fragment appearing in just 2 unrelated products -- HIGHER than
# 75.0 against "milk" itself (648 docs), because rapidfuzz's ratio
# rewards a same-length deletion over a same-length substitution for very
# short strings. Score threshold cannot separate these two cases: a
# genuine correction ("crakers"->"cracker", "جبنن"->"جبن") scores the
# identical 85.7.
#
# SPELLING_CORRECTION_MIN_TARGET_DOCS=3 (the correction target must
# appear in at least this many real products) is the signal that does
# separate them -- confirmed directly against the real index: every
# genuine correction found in testing has >=3 docs (lowest observed:
# "halloumii"->"halloumi" at exactly 3, "freekehh"->"freekeh" at 4), while
# the "milq"->"mil" false positive has only 2. This is a narrow, not a
# wide, margin (3 vs. 2) -- disclosed honestly, not claimed clean: a
# handful of confirmed remaining false positives exist where an unrelated
# but genuinely common real word coincidentally out-scores the intended
# target (e.g. "تشيس" (typo of تشيز/cheese) -> "تشويس" (Choice, a real
# brand, 90 docs), and "edammame" (typo of edamame, not itself in the
# vocabulary) -> "medamme" (foul medammes, a real, unrelated dish, 12
# docs)) -- the same class of inherent edit-distance limitation already
# accepted for brand/category matching and the vector-fallback floor, not
# something a threshold/floor tweak can fully close.
SPELLING_CORRECTION_THRESHOLD = 80
SPELLING_CORRECTION_MIN_TARGET_DOCS = 3


def correct_unmatched_terms(index: SearchIndex, query_text: str | None) -> str | None:
    """For each query word with ZERO literal matches in index.postings,
    try to fuzzy-correct it to a real, sufficiently-common catalog word.
    Words that already match something literally are never touched or
    re-scored. Returns the corrected query text if at least one word was
    actually changed, else None (so the caller can tell "nothing to try"
    apart from "corrected text happens to equal the input")."""
    tokens = tokenize(query_text, index.base_freq)
    if not tokens:
        return None

    vocabulary = list(index.postings.keys())
    corrected_tokens: list[str] = []
    changed = False
    for token in tokens:
        if token in index.postings:
            corrected_tokens.append(token)
            continue
        match = process.extractOne(token, vocabulary, scorer=fuzz.ratio)
        if (
            match is not None
            and match[1] >= SPELLING_CORRECTION_THRESHOLD
            and len(index.postings[match[0]]) >= SPELLING_CORRECTION_MIN_TARGET_DOCS
        ):
            corrected_tokens.append(match[0])
            changed = True
        else:
            corrected_tokens.append(token)

    return " ".join(corrected_tokens) if changed else None


def bm25_score(index: SearchIndex, doc_id: int, query_tokens: list[str]) -> float:
    tf_doc = index.doc_term_freq.get(doc_id)
    if tf_doc is None:
        return 0.0
    doc_len = index.doc_len[doc_id]
    score = 0.0
    for token in set(query_tokens):
        tf = tf_doc.get(token, 0)
        if tf == 0:
            continue
        idf = index.idf(token)
        denom = tf + K1 * (1 - B + B * doc_len / (index.avg_doc_len or 1))
        score += idf * (tf * (K1 + 1)) / (denom or 1)
    return score


def corpus_cooccurrence_ids(index: SearchIndex, words: list[str]) -> set[int]:
    """Real product IDs whose own tokenization contains ALL of `words`
    together (set intersection over SearchIndex.postings). Word-order-
    agnostic on purpose: a real catalog title often states a well-
    attested phrase in a different order than a customer's query
    ("Olive Oil Extra Virgin" vs "Extra Virgin Olive Oil").

    The single shared low-level primitive for "does the catalog actually
    combine these words" -- src/search_engine/keyword_baseline.py's
    _corpus_cooccurrence_ids() (used by the out-of-catalog gate and fuzzy
    category/brand grounding) is a thin wrapper around this exact
    function, and src/search_adapter/adapter.py's fusion-skip check uses
    it directly. Lives here, not in keyword_baseline.py, specifically so
    adapter.py can use it without a circular import: keyword_baseline.py
    itself imports FROM src.search_adapter.adapter, so adapter.py
    importing back from keyword_baseline.py would be circular; ranking.py
    sits BELOW both (only src.search_engine.index/tokenizer), so either
    module can safely import from here."""
    ids: set[int] | None = None
    for w in words:
        toks = tokenize(w, index.base_freq)
        if not toks:
            return set()
        pids = index.postings.get(toks[0], set())
        ids = pids if ids is None else (ids & pids)
        if not ids:
            return set()
    return ids if ids is not None else set()


def rank(
    index: SearchIndex,
    candidate_ids: set[int],
    query_text: str | None,
) -> list[tuple[int, float]]:
    """Score and sort `candidate_ids` (already hard-filtered by Postgres --
    Stage 1). If query_text is empty, every candidate scores 0 and the
    original (Stage-1) order is preserved -- a pure filter query (category
    and/or price only, no free text) still returns all matches, just
    unranked, which is correct: there is no relevance signal to rank by."""
    query_tokens = tokenize(query_text, index.base_freq)
    if not query_tokens:
        return [(pid, 0.0) for pid in candidate_ids]

    # A free-text query must also FILTER, not just rank: restrict to docs
    # that actually contain at least one query token before scoring.
    # Without this, a query with zero real matches would still "rank" every
    # Stage-1 candidate (scoring them all 0) and report a nonzero
    # total_count/zero_result=False -- silently wrong per FR-012.
    text_matches = candidate_ids & index.candidates_for_tokens(query_tokens)
    # SKU isn't part of the name-token postings (it's a separate field), so
    # an exact-SKU query needs its own fallback into the match set.
    stripped_query = (query_text or "").strip()
    if stripped_query:
        text_matches |= {
            pid for pid in candidate_ids
            if index.doc_sku.get(pid) == stripped_query
        }

    scored: list[tuple[int, float]] = []
    query_token_set = set(query_tokens)
    for pid in text_matches:
        score = bm25_score(index, pid, query_tokens)
        sku = index.doc_sku.get(pid, "")
        if sku and sku in (query_text or ""):
            score += EXACT_SKU_BOOST
        url_key_tokens = index.doc_url_key_tokens.get(pid, set())
        if url_key_tokens:
            # Sprint 6 product-identity relevance task: scaled by how much
            # of THIS product's own url_key the query actually covers
            # (query-to-title precision), not a flat per-token add. A flat
            # add rewarded any product whose marketing-style slug happens
            # to spell out the query term exactly as much as one whose
            # ENTIRE slug is that term -- real measured symptom: a generic
            # single-word query ("watermelon") got the SAME +2.0 for a
            # slug that is 1/6 query-related ("vidal-candy-belts-with-
            # watermelon-100g") as it would for a slug that IS the query,
            # systematically outranking genuine direct-identity products
            # whose slug is a bare SKU number (0 tokens, 0 boost either
            # way) purely on this mismatched weighting. Precision-scaling
            # keeps the boost meaningful for an actual slug/name match
            # while no longer treating an incidental mention the same as
            # the product's own core identity.
            score += URL_KEY_TOKEN_BOOST * len(query_token_set & url_key_tokens) / len(url_key_tokens)
        scored.append((pid, score))

    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored
