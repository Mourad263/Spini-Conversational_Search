"""Sprint 6 effective-price sorting task: the filter-only (no query_text)
path in src/search_adapter/adapter.py's SearchAdapter.search() sorted by
raw `price`, ignoring special_price entirely -- a discounted product
sorted by its pre-discount price, and a genuinely unknown-price row
(price=special_price=0) sorted as the cheapest, ahead of every real price.
_price_sort_key() now reuses the SAME neutral effective-price rule
(src/pricing.py's effective_price() -- also used by src/checkout/
demo_order.py and mirroring src/search_engine/query.py's own SQL CASE
expression), with unknown prices always placed after every known price.
Production only ever sorts ascending (there is no descending caller), so
that is the only order tested here.

Two layers: A-B test _price_sort_key() directly with synthetic rows (exact
control over the task's own 100/50, 60/0, 0/0 shapes); C-E confirm the real
SearchAdapter wiring against real, previously-verified catalog rows (same
IDs as test_effective_price_filter.py) and that query_text relevance
ranking is untouched.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.pricing import effective_price  # noqa: E402
from src.search_adapter.adapter import FilterSet, SearchAdapter, _price_sort_key  # noqa: E402

# Real, previously-verified rows (same as test_effective_price_filter.py):
_DISCOUNTED_ID = 1130     # "Abdul Razeq Custards": price=209.95, special_price=199.95
_SENTINEL_ID = 13993      # "Kabochas - 1 Piece": price=9.50, special_price=0.00
_BOTH_ZERO_ID = 8631      # "Forno Buano Cake - 750G": price=0.00, special_price=0.00

_ROWS = [
    {"id": 1, "price": 100.0, "special_price": 50.0},   # effective 50 (discounted)
    {"id": 2, "price": 60.0, "special_price": 0.0},      # effective 60 (sentinel special_price)
    {"id": 3, "price": 0.0, "special_price": 0.0},       # unknown -- both fields 0
]


# --- A: ascending -- effective price 50 (discounted from 100) before 60 ---

def test_ascending_sorts_by_effective_price_not_raw_price():
    ordered = sorted(_ROWS, key=_price_sort_key)
    assert [r["id"] for r in ordered] == [1, 2, 3]


# --- B: unknown price (0/0) never sorts as cheapest -- comes after every
# known price ---

def test_unknown_price_sorts_after_known_prices():
    ordered = sorted(_ROWS, key=_price_sort_key)
    assert ordered[-1]["id"] == 3


# --- C: real SearchAdapter wiring -- discounted product (effective 199.95)
# sorts after the sentinel-special-price product (effective 9.50); use a
# wide filter and check RELATIVE order of known real rows ---

def test_real_search_adapter_orders_by_effective_price_ascending():
    result = SearchAdapter().search(FilterSet(price_min=0.01, price_max=1000.0), limit=30000, offset=0)
    ids_in_order = [p["id"] for p in result.products]
    assert _SENTINEL_ID in ids_in_order and _DISCOUNTED_ID in ids_in_order
    # sentinel's effective price (9.50) is far below the discounted
    # product's (199.95) -- must appear earlier in ascending order.
    assert ids_in_order.index(_SENTINEL_ID) < ids_in_order.index(_DISCOUNTED_ID)


# --- D: the real both-zero row is excluded from a bounded price query
# entirely (test_effective_price_filter.py's own established behavior --
# hard_filter itself never returns it here), so it cannot appear as
# "cheapest" in a filtered+sorted result at all ---

def test_real_unknown_price_row_absent_from_bounded_sorted_results():
    result = SearchAdapter().search(FilterSet(price_min=0.01, price_max=1000.0), limit=30000, offset=0)
    ids_in_order = [p["id"] for p in result.products]
    assert _BOTH_ZERO_ID not in ids_in_order


# --- E: the query_text (relevance-ranked) path is NOT price-sorted -- a
# meaningful, non-tautological control proving this task's price-sort
# path (which only ever runs when query_text is absent) never leaks into
# the relevance-ranking path. Uses effective_price() only as a plain
# value-reader for the assertion, never SearchAdapter's own _price_sort_key
# -- if relevance ranking had been silently replaced by price ordering,
# effective prices would come back non-decreasing; real BM25/vector
# relevance scores have no such correlation with price. ---

def test_query_text_path_is_not_price_sorted():
    result = SearchAdapter().search(FilterSet(query_text="milk"), limit=20, offset=0)
    assert len(result.products) > 1
    known_prices = [
        price for p in result.products
        if (price := effective_price(p["price"], p["special_price"])) is not None
    ]
    assert len(known_prices) > 1
    assert known_prices != sorted(known_prices)
