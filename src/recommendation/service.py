"""Production content-based recommendation service (Sprint 6
recommendation task), promoted from the proven offline prototype
(evaluation/recommendation_cluster_experiment.py,
evaluation/recommendation_filtered_prototype.py -- 9/9 acceptance
criteria there). This module owns no new filter/embedding logic: it
composes the project's EXISTING hard_filter() and embed() with a frozen,
pre-trained clustering artifact.

Responsibility boundary, same spirit as the LLM/validation split
elsewhere in this project: this service PROPOSES candidates from
semantic similarity; it never decides what is allowed -- hard_filter()
is the only authority for that, exactly the same function the main
search path uses. Recommendation is read-only over session state: it
never calls StateManager.apply() and never mutates the FilterSet it is
given.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import psycopg  # noqa: E402
from pgvector.psycopg import register_vector  # noqa: E402

from src.models.config import settings  # noqa: E402
from src.search_adapter.adapter import FilterSet  # noqa: E402
from src.search_engine.embeddings import embed  # noqa: E402
from src.search_engine.keyword_baseline import _corpus_cooccurrence_ids  # noqa: E402
from src.search_engine.query import hard_filter  # noqa: E402

KMEANS_ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts" / "product_kmeans_128.joblib"
# Frozen offline artifact (evaluation/recommendation_cluster_experiment.py):
# MiniBatchKMeans, k=128, random_state=42, fit once on the existing 25,881
# normalized 768-d product embeddings (ibm-granite/granite-embedding-278m-
# multilingual). k=128 was selected over 64/256 by measured cluster
# coherence/balance, not silhouette score alone -- see that experiment's
# report. Copied here (not imported from evaluation/) so production code
# has no runtime dependency on the evaluation/ tree; NEVER retrained here.

WEIGHT_QUERY = 0.5
WEIGHT_PROFILE = 0.5
# V1 prototype weights (evaluation/recommendation_filtered_prototype.py) --
# NOT claimed optimal, kept in one obvious place so a later evaluation pass
# can measure/sweep them without touching the ranking logic itself.


class RecommendationService:
    """Content-based recommendation candidate generator.

    Candidate flow (fixed, matches the proven offline prototype):
      relevant K=128 cluster(s) of the current search results
        -> exclude products already shown
        -> hard_filter() with the session's REAL active filters (the only
           enforcement point -- clustering/ranking never override it)
        -> rank survivors by WEIGHT_QUERY*query-cosine + WEIGHT_PROFILE*profile-cosine
        -> top N real product rows (same shape as hard_filter()'s rows, so
           callers can reuse the existing ProductOut conversion unchanged)
    """

    def __init__(self) -> None:
        self.model = joblib.load(KMEANS_ARTIFACT_PATH)  # frozen, never refit
        self._pids, self._embeddings, self._pid_to_idx = self._load_catalog_embeddings()
        self._labels = self.model.predict(self._embeddings)  # cheap: reuses frozen centroids

    @staticmethod
    def _load_catalog_embeddings() -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
        url = settings.database_url.replace("postgresql+psycopg://", "postgresql://")
        with psycopg.connect(url) as conn:
            register_vector(conn)  # must run before the cursor that uses it (psycopg3
            # cursors snapshot the adapter map at creation time)
            cur = conn.cursor()
            cur.execute("SELECT product_id, embedding FROM product_embeddings ORDER BY product_id")
            rows = cur.fetchall()
        pids = np.array([r[0] for r in rows], dtype=np.int64)
        embeddings = np.array([r[1].to_numpy() for r in rows], dtype=np.float32)
        pid_to_idx = {int(pid): i for i, pid in enumerate(pids)}
        return pids, embeddings, pid_to_idx

    def recommend(self, query_text: str | None, search_result_ids: list[int],
                   active_filters: FilterSet, limit: int = 5) -> list[dict]:
        """Real product rows only (hard_filter()'s own dict shape --
        id/name_en/name_ar/price/special_price/brand_normalized/url_key/
        sku), never more than `limit`, never one already in
        `search_result_ids`, never one that fails `active_filters`.

        Returns [] whenever there is no usable evidence (no valid result
        ids, e.g. an empty search or all-invalid ids) or nothing survives
        the cluster+filter narrowing. Both are correct, SAFE outcomes --
        e.g. an active lactose-free constraint whose only real products
        are already shown -- never a signal to relax the filter or fall
        back to an unconstrained recommendation."""
        shown_idx = [self._pid_to_idx[pid] for pid in search_result_ids if pid in self._pid_to_idx]
        if not shown_idx:
            return []

        profile = self._embeddings[shown_idx].mean(axis=0)
        profile = profile / (np.linalg.norm(profile) + 1e-9)

        query_vec = None
        if query_text:
            qv = np.array(embed(query_text), dtype=np.float32)
            query_vec = qv / (np.linalg.norm(qv) + 1e-9)

        relevant_clusters = set(int(self._labels[i]) for i in shown_idx)
        cluster_mask = np.isin(self._labels, list(relevant_clusters))
        shown_set = set(shown_idx)
        candidate_idx = np.array([i for i in np.where(cluster_mask)[0] if i not in shown_set])
        if len(candidate_idx) == 0:
            return []

        # The ONLY filter-enforcement point: the project's real hard_filter(),
        # never reimplemented, never bypassed. Clustering only ever narrows
        # candidates before this; it can never widen past what hard_filter allows.
        filtered_rows = hard_filter(active_filters.category, active_filters.price_min,
                                     active_filters.price_max, active_filters.brand)
        filtered_by_id = {row["id"]: row for row in filtered_rows}

        # hard_filter() only enforces STRUCTURED facets (category/price/
        # brand) -- it has no concept of the free-text product identity in
        # query_text. When neither category nor brand ever grounded for the
        # active search (a real, common case -- see validate.py), the active
        # PRODUCT IDENTITY lives ONLY in query_text, and without reusing it
        # here too, "candidates" collapses to "anything in the price range,"
        # silently abandoning the current shopping topic (current-context-
        # recommendations fix). Reuses the SAME corpus-postings evidence
        # orchestrator.py's out-of-catalog gate already relies on -- no new
        # vocabulary system, no query-specific rule. Only ever NARROWS an
        # already-hard_filter()-passing set; never widens past it, and never
        # zeroes it out on inconclusive (empty) text evidence -- a query
        # that reached this point already cleared the out-of-catalog gate.
        if active_filters.category is None and active_filters.brand is None and query_text:
            text_ids = _corpus_cooccurrence_ids(query_text.split())
            if text_ids:
                filtered_by_id = {pid: row for pid, row in filtered_by_id.items() if pid in text_ids}

        candidate_idx = np.array([i for i in candidate_idx if int(self._pids[i]) in filtered_by_id])
        if len(candidate_idx) == 0:
            return []

        profile_sims = self._embeddings[candidate_idx] @ profile
        if query_vec is not None:
            query_sims = self._embeddings[candidate_idx] @ query_vec
            combined = WEIGHT_QUERY * query_sims + WEIGHT_PROFILE * profile_sims
        else:
            combined = profile_sims

        order = np.argsort(-combined)[:limit]
        return [filtered_by_id[int(self._pids[candidate_idx[o]])] for o in order]


_service: RecommendationService | None = None


def get_recommendation_service() -> RecommendationService:
    """Singleton: the KMeans artifact and the full catalog embedding
    matrix are loaded once per process and reused for every call -- never
    reloaded or refit per request."""
    global _service
    if _service is None:
        _service = RecommendationService()
    return _service
