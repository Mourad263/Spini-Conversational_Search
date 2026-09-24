"""Stage 1 of the hybrid search engine: Postgres hard filters only, no
ranking, no LIMIT/OFFSET (research.md §2). Category is expanded through
canonical_category_id to every raw record that resolves to it (FR-019).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import psycopg  # noqa: E402

from src.models.config import settings  # noqa: E402


def _connect() -> psycopg.Connection:
    url = settings.database_url.replace("postgresql+psycopg://", "postgresql://")
    return psycopg.connect(url)


def hard_filter(
    category_slug: str | None,
    price_min: float | None,
    price_max: float | None,
    brand: str | None,
) -> list[dict]:
    """Returns every product row matching the structured filters -- category
    (canonical slug, expanded to all equivalent raw category records),
    price range, and brand (exact canonical identity against the
    canonicalized brand_normalized column, case-insensitive -- see the
    predicate below for why). No ranking, no LIMIT: Stage 2 (ranking.py)
    slices after sorting, so pagination never silently drops a matching
    product ahead of a lower-ranked one still within the filters (FR-012's
    "don't silently loosen/drop matches" principle, applied to pagination)."""
    where = ["1=1"]
    params: list = []

    # Sprint 6 effective-price consistency task: the same rule display and
    # checkout now use (src/checkout/demo_order.py's _unit_price(),
    # frontend/src/components/ProductCard.jsx's hasSpecial) -- special_price
    # is a real discount only when it's positive (0 is this catalog's own
    # "no special price recorded" sentinel, proven directly against the DB:
    # special_price > price never occurs; special_price == price is the
    # normal no-discount shape), otherwise the regular price applies.
    effective_price = "(CASE WHEN p.special_price > 0 THEN p.special_price ELSE p.price END)"

    if price_min is not None or price_max is not None:
        # A product with no valid positive effective price (both price and
        # special_price are 0 -- a real ~78-row ingestion gap, not a
        # genuine free item) is UNKNOWN, not "under any bound" -- it must
        # never satisfy a bounded price query merely because 0 <= max.
        where.append(f"{effective_price} > 0")

    if category_slug is not None:
        where.append("""
            EXISTS (
                SELECT 1 FROM product_category pc
                JOIN category c ON c.id = pc.category_id
                JOIN canonical_category cc ON cc.id = c.canonical_category_id
                WHERE pc.product_id = p.id AND cc.slug = %s
            )
        """)
        params.append(category_slug)

    if price_min is not None:
        where.append(f"{effective_price} >= %s")
        params.append(price_min)

    if price_max is not None:
        where.append(f"{effective_price} <= %s")
        params.append(price_max)

    if brand is not None:
        # Exact canonical identity, not substring: by the time brand reaches
        # here it is always an already-resolved canonical_brand.slug (the
        # only production call site is parse_keyword_query's fuzzy
        # resolver), so a substring match only risks false-positive
        # collisions between distinct real brands (confirmed: brand="aqua"
        # matched 4 unrelated brands sharing the substring "aqua";
        # brand="fa" matched 21 unrelated brands sharing "fa"). LOWER() on
        # the incoming param only (never a wildcard) keeps this exact, not
        # substring -- brand_normalized itself is always already lowercase
        # by construction (brand_normalization.py's slugify()), confirmed
        # via a live scan (0 mixed-case rows).
        where.append("p.brand_normalized = LOWER(%s)")
        params.append(brand)

    sql = f"""
        SELECT p.id, p.name_en, p.name_ar, p.price, p.special_price,
               p.brand_normalized, p.url_key, p.sku
        FROM product p
        WHERE {' AND '.join(where)}
    """

    with _connect() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def product_by_id(product_id: int) -> dict | None:
    """Final UI task: server-authoritative single-product lookup by primary
    key -- the SAME row shape hard_filter() returns. Reused by the
    deterministic, button-driven cart-mutation endpoint (src/scenario/
    cart_mutation.py) so a UI button's product_id is always resolved
    against real catalog data; name/price are never trusted from the
    client. Returns None when the id doesn't exist."""
    sql = """
        SELECT p.id, p.name_en, p.name_ar, p.price, p.special_price,
               p.brand_normalized, p.url_key, p.sku
        FROM product p
        WHERE p.id = %s
    """
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(sql, (product_id,))
        row = cur.fetchone()
        if row is None:
            return None
        cols = [d.name for d in cur.description]
        return dict(zip(cols, row))
