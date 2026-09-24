"""Stage 2b of the search engine (Sprint 1b Part A): dense/semantic
retrieval over product_embeddings, restricted to the Stage-1 candidate set
(same restriction BM25's text_matches already applies -- see ranking.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import psycopg  # noqa: E402

from src.models.config import settings  # noqa: E402
from src.search_engine.embeddings import embed  # noqa: E402


def _connect() -> psycopg.Connection:
    url = settings.database_url.replace("postgresql+psycopg://", "postgresql://")
    return psycopg.connect(url)


VECTOR_TOP_N = 100  # exposed so SearchAdapter can tell whether this LIMIT
# would actually filter anything for a given candidate pool size (see
# adapter.py) -- when candidate_ids is no bigger than this, vector_rank
# returns literally the whole pool, not a real top-N subset.

# Cosine-distance gate for the VECTOR-ONLY fallback (BM25 found nothing).
# Measured directly over the real catalog on 23 queries that actually take
# that path -- best (nearest) distance per query, top result checked by hand:
#     GENUINE rescues (13): 0.2329 "orangee"->Oranges .. 0.2976 "milq"->
#         Milka Milk Chocolate. Others in between: "lemonn" 0.2375,
#         "sardin" 0.2467, "bananna" 0.2469, "cheez" 0.2563, "ketchupp"
#         0.2581, "olivee" 0.2653, "popcornn" 0.2673, "coockies" 0.2833,
#         "joghurt" 0.2877, "wafflee" 0.2958.
#     GARBAGE / NONSENSE (10): 0.3046 "margarin"->Tomato Paste, 0.3167
#         "لانشن"->Body Lotion, 0.3178 "زذأزذأزذأ", 0.3283 "حليبب"->Lip
#         Balm, 0.3359 "tomatto"->Tornado coffee maker, 0.3442
#         "qwzxjklpv", 0.3439 "عصيرر"->Hair Oil, 0.3541 "جبنن"->Ghee,
#         0.3670 "juise"->Glass Cleaner, 0.3703 "tunna"->Tornado dishwasher.
# The groups nearly separate: worst genuine 0.2976 < best garbage 0.3046.
# 0.31 sits above that boundary deliberately -- it keeps all 13 genuine
# rescues with a 0.0124 margin on the case this path exists for, at the
# cost of letting the single marginal "margarin" (0.3046) through. A
# tighter 0.30 would also catch that one, but leaves only 0.0024 of margin
# above "milq", so an unseen genuine rescue landing just past it would be
# silently killed -- the worse failure of the two. NOT a clean, wide gap;
# it is a narrow one chosen to fail on the permissive side. Still far
# better separated than the brand-filtered flooding case (adapter.py),
# where genuine and irrelevant distances were fully interleaved and NO
# cutoff could have separated them at all.
VECTOR_FALLBACK_MAX_DISTANCE = 0.31


def vector_rank(
    candidate_ids: set[int], query_text: str, top_n: int = VECTOR_TOP_N,
    max_distance: float | None = None, per_result_max_distance: float | None = None,
) -> list[int]:
    """Embed `query_text` and return up to `top_n` candidate product ids
    ordered by ascending cosine distance (nearest first). Restricted to
    `candidate_ids` (the same Stage-1 hard-filtered set BM25 ranks within)
    so category/price/brand filters apply identically regardless of which
    retrieval path (keyword or semantic) found a product.

    `max_distance` gates the WHOLE result set on the single nearest match:
    if even the closest candidate is farther than it, nothing here is
    actually related to the query and [] is returned rather than a
    confidently-wrong "closest anyway" list. It deliberately does NOT
    filter result-by-result -- the distances within one genuinely-matching
    query's own results span well past any usable cutoff (real example:
    "milq" runs 0.2976 to 0.34+ across its 40 real milk results), so
    per-result filtering would gut a query the gate has already judged
    trustworthy. Left None by callers that fuse with BM25 for a query
    where every word already has its own literal corpus presence.

    `per_result_max_distance` is the OPPOSITE shape -- a per-row cutoff
    (this one) for the fusion path specifically, confirmed necessary live:
    for "turkey" (real BM25 matches on their own, so `max_distance` is
    never applied), vector_rank's un-gated top-100 pulled in "Turkish
    Coffee Maker" appliances purely by embedding proximity (0.357+,
    already past VECTOR_FALLBACK_MAX_DISTANCE=0.31) -- BM25 having ITS
    OWN evidence does not make every one of vector's top-100 additions
    trustworthy too. Genuine fusion rescues stay comfortably inside 0.31
    (confirmed: "cookies milq"'s real milk-related additions run
    0.21-0.26), so this never costs real recall, only unrelated tail
    noise."""
    if not candidate_ids:
        return []

    query_vec = embed(query_text)
    ids = list(candidate_ids)

    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT product_id, embedding <=> %s::vector AS distance
            FROM product_embeddings
            WHERE product_id = ANY(%s)
            ORDER BY distance
            LIMIT %s
            """,
            (query_vec, ids, top_n),
        )
        rows = cur.fetchall()

    if not rows:
        return []
    if max_distance is not None and rows[0][1] > max_distance:
        return []
    if per_result_max_distance is not None:
        rows = [row for row in rows if row[1] <= per_result_max_distance]
    return [row[0] for row in rows]


def reciprocal_rank_fusion(
    ranked_lists: list[list[int]], k: int = 60
) -> list[tuple[int, float]]:
    """Combine multiple ranked id lists into one fused ranking. For each
    list, a doc at position i (0-indexed) contributes 1/(k + i + 1); a doc
    appearing in more than one list accumulates contributions from each.
    k=60 is the standard RRF constant from the original paper (Cormack et
    al. 2009) -- picked because it's well-established, not tuned for this
    corpus. Returns (product_id, fused_score) sorted descending."""
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for i, pid in enumerate(ranked):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + i + 1)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
