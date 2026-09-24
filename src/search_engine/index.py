"""In-memory token index + BM25 corpus statistics (research.md §2).

Built once over the FULL product corpus (not per-query, not per-filtered-
subset) so relevance scores stay stable across a session's follow-up turns
-- recomputing IDF/avg-doc-length per query would make results shift in
ways that work against multi-turn refinement (FR-002/FR-003).
"""

from __future__ import annotations

import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import psycopg  # noqa: E402

from src.models.config import settings  # noqa: E402
from src.search_engine.tokenizer import _raw_tokens, tokenize  # noqa: E402


@dataclass
class SearchIndex:
    postings: dict[str, set[int]] = field(default_factory=lambda: defaultdict(set))
    doc_term_freq: dict[int, Counter] = field(default_factory=dict)
    doc_len: dict[int, int] = field(default_factory=dict)
    doc_sku: dict[int, str] = field(default_factory=dict)
    doc_url_key_tokens: dict[int, set[str]] = field(default_factory=dict)
    n_docs: int = 0
    avg_doc_len: float = 0.0
    # Corpus-wide pre-stem English token frequency, built once over the
    # whole corpus (Pass 1 below) -- lets tokenize() disambiguate the
    # genuinely spelling-ambiguous -es/-ies plural suffixes (see
    # tokenizer._trim_plural) by checking which candidate base form is
    # actually attested elsewhere in real product names. Query-time
    # tokenization (ranking.py) reuses this SAME Counter so a query and
    # the documents it's scored against are stemmed identically.
    base_freq: Counter = field(default_factory=Counter)

    def idf(self, token: str) -> float:
        df = len(self.postings.get(token, ()))
        if df == 0:
            return 0.0
        return math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)

    def candidates_for_tokens(self, tokens: list[str]) -> set[int]:
        candidates: set[int] = set()
        for t in tokens:
            candidates |= self.postings.get(t, set())
        return candidates

    @classmethod
    def build_from_db(cls) -> "SearchIndex":
        idx = cls()
        url = settings.database_url.replace("postgresql+psycopg://", "postgresql://")
        with psycopg.connect(url) as conn, conn.cursor() as cur:
            cur.execute("SELECT id, name_en, name_ar, url_key, sku FROM product")
            rows = cur.fetchall()

        # Pass 1: corpus-wide pre-stem token frequency (English only --
        # Arabic tokens never go through suffix trimming, digit-only
        # tokens can't disambiguate anything). Needed before Pass 2 can
        # tokenize any single document consistently with the rest of the
        # corpus.
        base_freq: Counter = Counter()
        for _pid, name_en, name_ar, _url_key, _sku in rows:
            for norm, arabic in _raw_tokens(name_en) + _raw_tokens(name_ar):
                if not arabic and not norm.isdigit():
                    base_freq[norm] += 1

        total_len = 0
        for pid, name_en, name_ar, url_key, sku in rows:
            # Sprint 6 product-identity relevance task: some catalog rows
            # have name_ar that is just name_en's own Latin text again
            # (untranslated data), not a real second-language mention -- a
            # plain concatenation then double-counted every shared token,
            # inflating BM25 term frequency for those rows alone (real
            # measured symptom: "Vidal Candy Belts With Watermelon" scored
            # tf=2 for "watermelon" purely from this duplication, ranking
            # above genuine watermelon-fruit rows with tf=1). Taking the
            # per-token MAX across the two fields instead of summing them
            # treats "the same word in both fields" as the one mention it
            # actually is; a token distinct to either field (the normal
            # case -- Arabic script tokens never collide with English
            # ones) is completely unaffected, since max(n, 0) == n.
            tf_en = Counter(tokenize(name_en, base_freq))
            tf_ar = Counter(tokenize(name_ar, base_freq))
            tf = Counter({
                token: max(tf_en.get(token, 0), tf_ar.get(token, 0))
                for token in tf_en.keys() | tf_ar.keys()
            })
            idx.doc_term_freq[pid] = tf
            idx.doc_len[pid] = sum(tf.values())
            idx.doc_sku[pid] = sku or ""
            idx.doc_url_key_tokens[pid] = set(tokenize((url_key or "").replace("-", " "), base_freq))
            for token in tf:
                idx.postings[token].add(pid)
            total_len += idx.doc_len[pid]

        idx.base_freq = base_freq
        idx.n_docs = len(idx.doc_len)
        idx.avg_doc_len = (total_len / idx.n_docs) if idx.n_docs else 0.0
        return idx


_singleton: SearchIndex | None = None


def get_index() -> SearchIndex:
    global _singleton
    if _singleton is None:
        _singleton = SearchIndex.build_from_db()
    return _singleton
