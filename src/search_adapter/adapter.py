"""SearchAdapter per contracts/search-adapter.md, composing the hybrid
engine's stages: Stage 1 (Postgres hard filters, query.py) -> Stage 2a
(hand-rolled BM25 ranking, ranking.py) + Stage 2b (dense/semantic vector
ranking, vector_search.py) -> Reciprocal Rank Fusion (Sprint 1b Part A).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.pricing import effective_price  # noqa: E402
from src.search_engine.index import get_index  # noqa: E402
from src.search_engine.query import hard_filter  # noqa: E402
from src.search_engine.ranking import corpus_cooccurrence_ids, correct_unmatched_terms, rank  # noqa: E402
from src.search_engine.vector_search import (  # noqa: E402
    VECTOR_FALLBACK_MAX_DISTANCE,
    VECTOR_TOP_N,
    reciprocal_rank_fusion,
    vector_rank,
)


def _price_sort_key(row: dict) -> tuple[bool, float]:
    """Sorts ascending by EFFECTIVE price (src/pricing.py's
    effective_price() -- the SAME rule price filtering (query.py's own
    effective_price CASE expression)/display/checkout already use), never
    raw `price` alone -- a discounted row must sort by what it actually
    costs, not its pre-discount price. An unknown price (neither field a
    real positive number, e.g. price=special_price=0) always sorts AFTER
    every known price -- never mistaken for the cheapest via a bare
    price=0."""
    price = effective_price(row["price"], row["special_price"])
    return (True, 0.0) if price is None else (False, price)


class InvalidFilterError(ValueError):
    """Raised when FilterSet is constructed with self-contradictory
    structured params (e.g. price_min > price_max). Deliberately NOT raised
    by the keyword-baseline's "between X and Y" text-parsing path, which
    auto-normalizes an inverted phrase before ever constructing a FilterSet
    -- this only guards direct structured construction, where a caller
    handing in genuinely malformed params should get a clear rejection, not
    a silently empty result indistinguishable from a valid empty range."""


@dataclass
class FilterSet:
    category: str | None = None      # a canonical_category.slug, or None
    price_min: float | None = None
    price_max: float | None = None
    brand: str | None = None         # a canonical_brand.slug -- matched via product.brand_normalized,
    # exact identity (case-insensitive), not substring (src/search_engine/query.py)
    query_text: str | None = None    # free-text search terms; NOT in the original contract
    # draft -- added during Sprint 1 implementation because System A (the
    # keyword baseline) fundamentally needs free-text search, which
    # structured filters alone can't express. See contracts/search-adapter.md.
    unresolved_exclusion: str | None = None  # Sprint 6: a free-from concept
    # (e.g. "vanilla") the user explicitly asked to exclude but that has no
    # real "X Free" catalog category to ground it to. NOT enforced as a
    # search filter -- this project has no general NOT-filter mechanism and
    # inventing one is out of scope. Carried through only so the response
    # layer (src/response/messages.py via orchestrator.py) can tell the user
    # the exclusion wasn't applied, instead of silently returning ordinary
    # results as if it had been honored.

    def __post_init__(self) -> None:
        if self.price_min is not None and self.price_max is not None \
           and self.price_min > self.price_max:
            raise InvalidFilterError(
                f"price_min ({self.price_min}) must not be greater than "
                f"price_max ({self.price_max})"
            )


@dataclass
class SearchResult:
    products: list[dict]   # corpus Products only (FR-011) -- never synthesized
    total_count: int
    zero_result: bool      # explicit, not inferred from an empty list downstream


class SearchAdapter:
    def search(self, filters: FilterSet, limit: int = 20, offset: int = 0) -> SearchResult:
        rows = hard_filter(filters.category, filters.price_min, filters.price_max, filters.brand)
        by_id = {row["id"]: row for row in rows}
        candidate_ids = set(by_id)

        if filters.query_text:
            index = get_index()
            original_ranked = rank(index, candidate_ids, filters.query_text)
            original_ids = [pid for pid, _score in original_ranked]

            # Sprint 6: try spelling correction UNCONDITIONALLY, before
            # deciding anything else -- not just when the whole query has
            # zero literal matches. correct_unmatched_terms only ever
            # touches a token that itself has zero literal matches, so its
            # corrected text can only find a candidate set that is a
            # superset of (or equal to) the original's, never a smaller
            # one. This is exactly what a multi-word query with ONE
            # misspelled term needs: "fresh chiken breast" used to never
            # reach correction at all, because "fresh"/"breast" already
            # matching made bm25_ids non-empty for the whole query, and
            # correction was previously gated on bm25_ids being entirely
            # empty.
            corrected_text = correct_unmatched_terms(index, filters.query_text)
            if corrected_text:
                bm25_ranked = rank(index, candidate_ids, corrected_text)
                bm25_ids = [pid for pid, _score in bm25_ranked]
            else:
                bm25_ranked, bm25_ids = original_ranked, original_ids

            if not original_ids and bm25_ids:
                # The ORIGINAL query matched nothing at all and correction
                # rescued the whole thing -- trust that directly, no vector
                # fusion, regardless of pool size: a confident correction
                # is not a noisier nearest-neighbour guess. Real symptom:
                # "لانشن" (typo of لانشون/luncheon) used to fall straight to
                # the vector floor and return nothing (correctly
                # suppressed, but the 90 real luncheon products were
                # sitting right there under a one-character typo).
                ranked = bm25_ranked
            elif bm25_ids and len(candidate_ids) <= VECTOR_TOP_N:
                # BM25 already found real literal matches, AND the
                # candidate pool is small enough that vector_rank's LIMIT
                # VECTOR_TOP_N would be a no-op -- it would return the
                # ENTIRE pool, not a real top-N subset, re-flooding in
                # every candidate BM25 had correctly excluded. Confirmed
                # directly: for "لانشون اطياب" (brand=atyab, 71
                # candidates), vector_rank returned all 71 regardless of
                # relevance (genuine luncheon-product distances
                # 0.459-0.497 fully interleaved with irrelevant same-brand
                # products, closest non-match 0.438 -- no distance
                # threshold could separate them either), so total_count
                # was always the brand's full catalog count instead of
                # the 9 genuine matches. Trust BM25 alone in that regime.
                ranked = bm25_ranked
            elif bm25_ids and len((filters.query_text or "").split()) >= 2 and \
                    corpus_cooccurrence_ids(index, (filters.query_text or "").split()):
                # Sprint 6 product-identity relevance task: a MULTI-word
                # query where a real product's own tokenization already
                # contains every word together (the same shared corpus-
                # evidence primitive the out-of-catalog gate uses, via
                # keyword_baseline.py's _corpus_cooccurrence_ids) -- the
                # query is itself a directly-attested catalog concept,
                # not a paraphrase needing semantic recovery (unlike
                # "cookies milq", where no real product combines
                # "cookies" with "milq" at all -- that case still falls
                # through to the fusion branch below). Deliberately NOT
                # applied to single-word queries too (a tempting
                # generalization that regressed two existing, confirmed
                # single-word fixes -- "cookie" and "dish" -- which rely
                # on vector fusion adding genuinely relevant items with
                # no literal token overlap at all; single-word ranking is
                # instead corrected directly in ranking.py/index.py
                # below). Fusing on an already-complete, directly-
                # attested multi-word concept only interleaves
                # semantically-nearby-but-wrong neighbours by RRF
                # position ahead of the strongest literal matches on
                # THIS query's own lexical evidence. "watermelon juice"
                # (no real product combines both words) deliberately
                # still reaches fusion below -- vector genuinely bridges
                # "juice" to real Drink products.
                ranked = bm25_ranked
            elif bm25_ids:
                # BM25 found real matches AND the candidate pool is large
                # enough (> VECTOR_TOP_N) that vector_rank's LIMIT is a
                # REAL filter, not a no-op -- confirmed directly: for
                # "cookies milq" (25,881 candidates, no structured
                # filter), vector_rank returned only 40 of them, 25 not
                # already found by BM25's 87 "cookies" matches, and those
                # 25 included genuinely relevant milk-typo rescues ("Milka
                # Milk Chocolate", "Mcvitie's ... Milk Chocolate Biscuit")
                # that BM25's "cookies" match alone could never surface.
                # Fusing here adds real recall without reintroducing the
                # flooding case above. BM25 having its OWN literal-match
                # evidence does not make every one of vector's top-100
                # additions trustworthy too, though -- real symptom: for
                # "turkey" (67 real literal matches), un-gated fusion
                # ranked "Turkish Coffee Maker" appliances into the
                # result purely by embedding proximity (0.357+ cosine
                # distance -- already past VECTOR_FALLBACK_MAX_DISTANCE).
                # The SAME already-calibrated per-result floor as the
                # vector-only fallback below keeps genuine fusion recall
                # (confirmed: "cookies milq"'s real milk-related additions
                # run 0.21-0.26, comfortably inside it) while dropping
                # tail noise no literal or semantic evidence supports.
                # Reciprocal Rank Fusion combines ranked lists by POSITION,
                # not raw score -- BM25 scores and cosine similarities live
                # on different, incomparable scales.
                vector_ids = vector_rank(
                    candidate_ids, filters.query_text, per_result_max_distance=VECTOR_FALLBACK_MAX_DISTANCE,
                )
                ranked = reciprocal_rank_fusion([bm25_ids, vector_ids])
            else:
                # BM25 found NOTHING even after the correction pass above
                # -- no single-token correction rescued it either (the
                # typo/cross-lingual rescue case vector search exists for,
                # e.g. "milq" -> milk, which correct_unmatched_terms
                # deliberately does NOT correct -- see ranking.py).
                # VECTOR_FALLBACK_MAX_DISTANCE gates the whole list on the
                # nearest match, so a query with nothing genuinely related
                # returns zero results instead of confidently-wrong ones
                # (real symptom: "لانشن" used to return body lotion;
                # measurements in vector_search.py). Real symptom the
                # correction pass above already fixes before we even get
                # here: "لانشن" (typo of لانشون/luncheon) used to fall
                # straight to this floor and return nothing (correctly
                # suppressed, but the 90 real luncheon products were
                # sitting right there under a one-character typo).
                vector_ids = vector_rank(candidate_ids, filters.query_text,
                                          max_distance=VECTOR_FALLBACK_MAX_DISTANCE)
                ranked = reciprocal_rank_fusion([vector_ids])
        else:
            # No text query: no BM25 score, no vector query to run either
            # (there is no query to embed). Filter-only candidates have no
            # relevance signal to rank by -- default to effective price
            # ascending (unknown prices last) as the least-surprising
            # order. Found live: this used to return Python set-iteration
            # order, which looked like nothing (not price, not name).
            ranked = [(pid, 0.0) for pid in candidate_ids]
            ranked.sort(key=lambda pair: _price_sort_key(by_id[pair[0]]))

        page = ranked[offset:offset + limit]
        products = [by_id[pid] for pid, _score in page]

        return SearchResult(
            products=products,
            total_count=len(ranked),
            zero_result=len(ranked) == 0,
        )
