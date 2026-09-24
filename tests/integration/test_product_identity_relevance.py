"""Sprint 6 product-identity relevance task: a query naming a real product
concept ("watermelon") must prefer products where the concept IS the
product over products where it is merely a flavor/scent/variant mention
("Watermelon Candy"), while an explicit derivative query ("watermelon
candy") must still retrieve the derivative fine. Two general, non-query-
specific fixes, against the real catalog/DB (no stubs -- this is a
relevance property of the real corpus, not something a fixture can
stand in for):

1. src/search_engine/index.py: doc term-frequency is now the per-token
   MAX across name_en/name_ar, not their sum -- some catalog rows store
   name_ar as a plain duplicate of name_en (untranslated data), which
   used to double-count every shared token's frequency for those rows
   only, inflating their BM25 score for no real reason.
2. src/search_engine/ranking.py: the url_key token boost is now scaled
   by what FRACTION of the product's own url_key the query covers
   (query-to-title precision), not a flat per-token add -- a marketing-
   style slug that happens to spell out the query term got the exact
   same boost as a slug that IS the query term.
3. src/search_adapter/adapter.py + ranking.py's corpus_cooccurrence_ids()
   (the single shared postings-intersection primitive also used by
   keyword_baseline.py's _corpus_cooccurrence_ids/the out-of-catalog
   gate): a multi-word query where a real product's own text already
   contains every word together skips vector fusion (nothing left to
   rescue); deliberately NOT extended to single-word queries, which
   regressed two pre-existing, confirmed fixes ("cookie"/"dish" -- see
   test_category_fuzzy.py) that rely on fusion adding genuinely relevant
   items with no literal token overlap at all.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.search_adapter.adapter import FilterSet, SearchAdapter  # noqa: E402
from src.search_engine.index import get_index  # noqa: E402
from src.search_engine.ranking import corpus_cooccurrence_ids  # noqa: E402


def _names(products: list[dict]) -> list[str]:
    return [p["name_en"] for p in products]


# --- A: simple watermelon query -- direct identity materially outranks
# derivative (flavor) mentions ---

def _is_direct_watermelon(name: str) -> bool:
    n = name.lower()
    return "watermelon" in n and not any(
        w in n for w in ("candy", "gum", "jelly", "drink", "sparkling", "marshmallow")
    )


def test_watermelon_direct_identity_outranks_derivative_mentions():
    # Real, measured pre-fix baseline for this exact live query shape
    # ("i want a kilo of watermelon under 100"): the first direct-identity
    # product ("Chile Watermelons") appeared at position 6, with only 3 of
    # the top 10 being direct identity (the rest candy/gum/jelly/drink).
    r = SearchAdapter().search(FilterSet(query_text="watermelon", price_max=100.0), limit=10, offset=0)
    top_10 = _names(r.products)
    first_direct_position = next(i for i, n in enumerate(top_10, start=1) if _is_direct_watermelon(n))
    direct_count = sum(1 for n in top_10 if _is_direct_watermelon(n))
    assert first_direct_position <= 3  # materially earlier than the pre-fix position 6
    assert direct_count >= 4  # materially more than the pre-fix 3 of 10


# --- B: watermelon + price -- price filter remains authoritative ---

def test_watermelon_price_filter_still_authoritative():
    # Sprint 6 effective-price consistency task: the filter now checks the
    # EFFECTIVE price (special_price when it's a real positive discount,
    # else price -- query.py's hard_filter), the same price checkout would
    # actually charge, so a product may legitimately pass via a special
    # price below the raw one.
    r = SearchAdapter().search(FilterSet(query_text="watermelon", price_max=20.0), limit=20, offset=0)
    assert r.products  # sanity: real matches exist under this bound

    def effective(p):
        return p["special_price"] if p["special_price"] and p["special_price"] > 0 else p["price"]

    assert all(effective(p) is None or effective(p) <= 20.0 for p in r.products)


# --- C: explicit derivative query remains retrievable ---

def test_explicit_derivative_queries_remain_retrievable():
    for derivative_query, expected_word in [
        ("watermelon candy", "candy"),
        ("watermelon gum", "gum"),
    ]:
        r = SearchAdapter().search(FilterSet(query_text=derivative_query), limit=5, offset=0)
        assert r.products
        assert any(expected_word in n.lower() for n in _names(r.products))


# --- D: two additional real-catalog generalization concepts ---

def test_strawberry_direct_identity_outranks_derivative_mentions():
    r = SearchAdapter().search(FilterSet(query_text="strawberry"), limit=6, offset=0)
    top = _names(r.products)
    direct = sum(1 for n in top if "strawberr" in n.lower() and "cream" not in n.lower()
                 and "scrub" not in n.lower() and "sauce" not in n.lower())
    assert direct >= 3


def test_mango_direct_identity_outranks_derivative_mentions():
    r = SearchAdapter().search(FilterSet(query_text="mango"), limit=6, offset=0)
    top = _names(r.products)
    direct = sum(1 for n in top if "mango" in n.lower() and "sauce" not in n.lower())
    assert direct >= 4


# --- E: direct product identity is not replaced by unrelated semantic
# neighbours -- the real fruit id set must be a subset of what's returned ---

def test_direct_identity_products_remain_present_not_dropped():
    r = SearchAdapter().search(FilterSet(query_text="watermelon"), limit=60, offset=0)
    names = _names(r.products)
    assert any(n == "Chile Watermelons" for n in names)
    assert any("Red Watermelons" in n for n in names)


# --- F: the multi-word cooccurrence signal itself is general, not a
# watermelon-specific rule ---

def test_cooccurrence_signal_is_general_not_watermelon_specific():
    index = get_index()
    assert corpus_cooccurrence_ids(index, ["potato", "chips"])
    assert corpus_cooccurrence_ids(index, ["watermelon", "candy"])
    assert not corpus_cooccurrence_ids(index, ["watermelon", "juice"])
    assert not corpus_cooccurrence_ids(index, ["laptop", "charger"])


# --- G: non-regression -- potato chips / laptop charger / car tires controls ---

def test_potato_chips_multi_word_identity_preserved():
    r = SearchAdapter().search(FilterSet(query_text="potato chips"), limit=6, offset=0)
    assert all("chip" in n.lower() for n in _names(r.products))


def test_out_of_catalog_controls_unaffected():
    from src.scenario.orchestrator import _lacks_catalog_evidence
    for query_text, expected in [
        ("bag of chips", True), ("laptop charger", True), ("car tires", True),
        ("potato chips", False), ("chips", False),
    ]:
        fs = FilterSet(query_text=query_text)
        assert _lacks_catalog_evidence(fs) is expected, query_text


# ---------------------------------------------------------------------
# Sprint 6 search+recommendation context task: a THIRD general fix --
# src/search_engine/vector_search.py's vector_rank() gained
# per_result_max_distance, applied by adapter.py's fusion branch using
# the SAME already-calibrated VECTOR_FALLBACK_MAX_DISTANCE=0.31 constant
# used for the vector-only fallback. Real symptom: "turkey" (67 literal
# BM25 matches -- real evidence a query means something) still let
# vector_rank's un-gated top-100 pull in "Turkish Coffee Maker"
# appliances purely by embedding proximity (0.357+, already past the
# floor); BM25 having its own evidence doesn't make every vector
# addition trustworthy too. Confirmed NOT to cost genuine fusion recall:
# "cookies milq"'s real milk-related additions run 0.21-0.26, comfortably
# inside the floor (test_hybrid_search.py's existing cookies-milq test
# still passes unchanged).
# ---------------------------------------------------------------------

def test_turkey_direct_identity_not_diluted_by_unrelated_semantic_neighbours():
    r = SearchAdapter().search(FilterSet(query_text="turkey", price_min=800.0), limit=20, offset=0)
    names = _names(r.products)
    assert any("Turkey" in n and "Lobes" in n for n in names)  # real turkey deli meat present
    assert not any("turkish" in n.lower() for n in names)  # no Turkish coffee makers/appliances
    assert not any("coffee" in n.lower() for n in names)


def test_turkey_direct_products_outrank_pet_food_mentions():
    r = SearchAdapter().search(FilterSet(query_text="turkey", price_min=800.0), limit=20, offset=0)
    names = _names(r.products)
    turkey_positions = [i for i, n in enumerate(names) if "Turkey" in n and "Dog" not in n and "Cat" not in n]
    pet_food_positions = [i for i, n in enumerate(names) if "Dog" in n or "Cat" in n]
    assert turkey_positions and pet_food_positions
    assert min(turkey_positions) < min(pet_food_positions)  # real turkey ranks ahead of pet food


def test_turkey_dog_food_derivative_query_remains_retrievable():
    r = SearchAdapter().search(FilterSet(query_text="turkey dog food"), limit=5, offset=0)
    assert any("dog" in n.lower() and "turkey" in n.lower() for n in _names(r.products))


def test_bag_of_chips_snack_identity_dominates_after_price_filter():
    r = SearchAdapter().search(FilterSet(query_text="chips", price_min=20.0), limit=15, offset=0)
    names = _names(r.products)
    assert sum(1 for n in names if "chip" in n.lower()) >= 12  # snack chips clearly dominate


def test_explicit_chocolate_chips_query_remains_retrievable():
    r = SearchAdapter().search(FilterSet(query_text="chocolate chips"), limit=5, offset=0)
    assert any("chocolate" in n.lower() and "chip" in n.lower() for n in _names(r.products))
