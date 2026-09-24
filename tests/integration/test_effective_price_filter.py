"""Sprint 6 effective-price consistency task: hard_filter()'s price_min/
price_max predicates must use the SAME effective-price rule now used by
checkout (src/checkout/demo_order.py's _unit_price()) and the frontend
(ProductCard.jsx) -- special_price only when it's a real, positive
discount, otherwise the regular price; a row where BOTH fields are 0 (a
real ~78-row ingestion gap) is unknown, not "free," and must never satisfy
a bounded price query. Against the real DB/catalog -- these are real,
previously-verified rows, not synthetic fixtures.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.checkout.demo_order import ProductSnapshot, _unit_price  # noqa: E402
from src.search_engine.query import hard_filter  # noqa: E402

# Real, previously-verified rows:
_DISCOUNTED_ID = 1130     # "Abdul Razeq Custards": price=209.95, special_price=199.95
_SENTINEL_ID = 13993      # "Kabochas - 1 Piece": price=9.50, special_price=0.00
_BOTH_ZERO_ID = 8631      # "Forno Buano Cake - 750G": price=0.00, special_price=0.00


def _ids(rows: list[dict]) -> set[int]:
    return {r["id"] for r in rows}


# --- A: a genuine discount is used for price_max (effective 199.95, not
# raw 209.95) ---

def test_discounted_product_included_under_effective_price_max():
    rows = hard_filter(None, None, 200.0, None)
    assert _DISCOUNTED_ID in _ids(rows)  # 199.95 <= 200, though raw price (209.95) is not


def test_discounted_product_excluded_when_effective_price_exceeds_max():
    rows = hard_filter(None, None, 190.0, None)
    assert _DISCOUNTED_ID not in _ids(rows)  # 199.95 > 190


# --- B: price_min also uses the effective price, not the raw one ---

def test_discounted_product_excluded_from_price_min_above_effective_price():
    # Raw price (209.95) would satisfy price_min=205, but the customer
    # actually pays 199.95 -- which does not.
    rows = hard_filter(None, 205.0, None, None)
    assert _DISCOUNTED_ID not in _ids(rows)


def test_discounted_product_included_in_price_min_below_effective_price():
    rows = hard_filter(None, 150.0, None, None)
    assert _DISCOUNTED_ID in _ids(rows)


# --- C/D: special_price=0 sentinel falls back to regular price in the filter ---

def test_sentinel_special_price_falls_back_to_regular_price_in_max_filter():
    rows = hard_filter(None, None, 20.0, None)
    assert _SENTINEL_ID in _ids(rows)  # regular 9.50 <= 20


def test_sentinel_special_price_excluded_when_regular_price_exceeds_max():
    rows = hard_filter(None, None, 5.0, None)
    assert _SENTINEL_ID not in _ids(rows)  # regular 9.50 > 5


# --- E: both price and special_price are 0 -- unknown, never "under any bound" ---

def test_both_zero_row_excluded_from_bounded_price_max_query():
    rows = hard_filter(None, None, 1000.0, None)
    assert _BOTH_ZERO_ID not in _ids(rows)  # NOT included just because 0 <= 1000


def test_both_zero_row_excluded_from_price_min_query_too():
    rows = hard_filter(None, 0.0, None, None)
    assert _BOTH_ZERO_ID not in _ids(rows)


# --- F: no price filter at all -- existing (no-price-constraint) behavior preserved ---

def test_both_zero_row_still_present_without_any_price_filter():
    rows = hard_filter(None, None, None, None)
    assert _BOTH_ZERO_ID in _ids(rows)


# --- G: checkout's effective unit price agrees with the filter's effective price ---

def test_checkout_unit_price_matches_filter_effective_price_for_same_product():
    snapshot = ProductSnapshot(id=_DISCOUNTED_ID, name_en="x", name_ar="x", price=209.95, special_price=199.95)
    assert float(_unit_price(snapshot)) == 199.95
    # the same 199.95 is exactly the boundary hard_filter's price_max used above
    assert _DISCOUNTED_ID in _ids(hard_filter(None, None, 199.95, None))
    assert _DISCOUNTED_ID not in _ids(hard_filter(None, None, 199.94, None))
