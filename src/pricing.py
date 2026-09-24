"""THE one place "effective price" is decided in Python -- deliberately
dependency-free (no checkout, no search, no DB) so neither domain depends
on the other: src/checkout/demo_order.py (order pricing) and
src/search_adapter/adapter.py (price sorting) both import this SAME rule
instead of each re-deriving it. Mirrors src/search_engine/query.py's own
SQL CASE expression used for price_min/price_max filtering (effective_price
= "(CASE WHEN p.special_price > 0 THEN p.special_price ELSE p.price
END)") -- same rule, two representations (SQL inside the DB query, Python
for values already fetched), never meant to drift apart.
"""

from __future__ import annotations


def effective_price(price: float | None, special_price: float | None) -> float | None:
    """Sprint 6 zero-price semantics task: special_price is NEVER null in
    this catalog (confirmed directly against the real DB -- 0 of 25,881
    rows), so a plain "special_price if not None else price" check always
    chose special_price, unconditionally. The source data uses
    special_price=0 as a "no special price recorded" SENTINEL for a real
    subset of rows (151 total; concentrated in fresh produce, e.g.
    "Kabochas" -- price=9.95, special_price=0.00), never a genuine
    free/zero sale price (special_price > price occurs in 0 rows;
    special_price == price in 25,026 rows is the normal "no discount"
    shape). Only a POSITIVE special_price is ever a real discount
    (search/recommendation display never compute one; they pass the raw
    price/special_price fields through unchanged, so frontend/
    ProductCard.jsx's own hasSpecial check needed the identical
    `special_price > 0` correction to stay consistent with this rule).
    Returns None when NEITHER field is a real positive price -- a
    genuinely unknown price, never fabricated as free/zero."""
    raw = special_price if special_price is not None and special_price > 0 else price
    return raw if raw is not None and raw > 0 else None
