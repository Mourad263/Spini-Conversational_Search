"""Adversarial input-boundary tests for the real HTTP endpoint. Found and
fixed one real gap here: `q` had no length validation actually wired in
(a matching Pydantic model existed in schemas.py but the route never used
it) -- fixed in routes.py via FastAPI's Query(..., min_length=..., ...).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi.testclient import TestClient  # noqa: E402

from src.api.routes import app  # noqa: E402
from src.search_adapter.adapter import FilterSet, InvalidFilterError, SearchAdapter  # noqa: E402

client = TestClient(app)
adapter = SearchAdapter()


def test_empty_query_rejected():
    r = client.get("/search/keyword", params={"q": ""})
    assert r.status_code == 422


def test_oversized_query_rejected():
    r = client.get("/search/keyword", params={"q": "chocolate " * 500})  # 5000 chars
    assert r.status_code == 422


def test_inverted_price_phrase_auto_normalized_not_error():
    r = client.get("/search/keyword", params={"q": "chicken between 100 and 20"})
    assert r.status_code == 200
    data = r.json()
    assert data["price_min"] == 20.0
    assert data["price_max"] == 100.0  # normalized, not left inverted


def test_sql_injection_shaped_input_is_safe():
    payloads = [
        "chocolate'; DROP TABLE product; --",
        'chocolate" OR "1"="1',
        "1' OR '1'='1",
    ]
    for payload in payloads:
        r = client.get("/search/keyword", params={"q": payload})
        assert r.status_code == 200  # never a 500, never an unhandled DB error
    # the table must still be fully intact after all three payloads
    result = adapter.search(FilterSet(), limit=1)
    assert result.total_count == 25881


def test_negative_price_min_is_a_noop_not_an_error():
    # Sprint 6 effective-price consistency task: ANY bounded price query
    # now also requires a valid positive effective price (query.py's
    # hard_filter), excluding the 78 real rows where both price and
    # special_price are 0 (an ingestion gap, not a genuine free price) --
    # those are "unknown," not "matches every bound including negative
    # ones." 25881 - 78 = 25803 is every OTHER real product.
    result = adapter.search(FilterSet(price_min=-50), limit=1)
    assert result.total_count == 25803


def test_genuinely_inverted_range_is_rejected_not_silently_empty():
    """price_min > price_max at the FilterSet layer (the keyword parser
    always auto-normalizes 'between X and Y', so this can't happen through
    it -- constructed directly). Previously degraded to a silent empty
    result indistinguishable from a genuinely-valid empty range; now
    rejected explicitly at construction, same spirit as the other
    boundary validations."""
    import pytest

    with pytest.raises(InvalidFilterError):
        FilterSet(price_min=500, price_max=10)
