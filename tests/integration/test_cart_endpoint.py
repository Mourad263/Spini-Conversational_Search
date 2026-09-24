"""Final UI + deterministic cart backend task: a real, hermetic-of-the-LLM
end-to-end walkthrough of POST /session/{session_id}/cart/item -- real
FastAPI app, real database (product_by_id()), real SessionActivityStore
singleton (the SAME one src/scenario/orchestrator.py's chat path uses).
Zero Groq/OpenAI calls anywhere in this file -- the endpoint under test
never reaches handle_message()/understand() at all.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi.testclient import TestClient  # noqa: E402

from src.api.routes import app  # noqa: E402
from src.scenario.orchestrator import get_activity_store  # noqa: E402
from src.scenario.session_activity import CartLine, DisplayedProduct  # noqa: E402

client = TestClient(app)

# Real catalog row, id confirmed live earlier this session.
_PEPSI_970_ID = 5293


def _reset(session_id: str) -> None:
    get_activity_store().reset(session_id)


def test_add_new_product_by_delta_resolves_against_real_catalog():
    session_id = "cart-endpoint-add"
    _reset(session_id)

    r = client.post(f"/session/{session_id}/cart/item", json={"product_id": _PEPSI_970_ID, "delta": 1})
    assert r.status_code == 200
    body = r.json()
    assert body["checkout_invalidated"] is False
    lines = body["cart"]["lines"]
    assert len(lines) == 1
    assert lines[0]["id"] == _PEPSI_970_ID
    assert lines[0]["quantity"] == 1
    assert lines[0]["name_en"]  # real catalog name, never client-supplied
    assert lines[0]["unit_price"] > 0  # real server-computed price
    assert body["cart"]["total"] == lines[0]["line_total"]


def test_increment_and_decrement_use_existing_cart_semantics():
    session_id = "cart-endpoint-stepper"
    _reset(session_id)

    client.post(f"/session/{session_id}/cart/item", json={"product_id": _PEPSI_970_ID, "delta": 1})
    r2 = client.post(f"/session/{session_id}/cart/item", json={"product_id": _PEPSI_970_ID, "delta": 1})
    assert r2.json()["cart"]["lines"][0]["quantity"] == 2

    r3 = client.post(f"/session/{session_id}/cart/item", json={"product_id": _PEPSI_970_ID, "delta": -1})
    assert r3.json()["cart"]["lines"][0]["quantity"] == 1

    # decrementing to zero removes the line via the SAME server cart
    # semantics as a chat "remove" -- never a lingering qty=0 row
    r4 = client.post(f"/session/{session_id}/cart/item", json={"product_id": _PEPSI_970_ID, "delta": -1})
    assert r4.json()["cart"]["lines"] == []


def test_decrement_on_absent_product_is_a_harmless_no_op():
    session_id = "cart-endpoint-noop"
    _reset(session_id)
    r = client.post(f"/session/{session_id}/cart/item", json={"product_id": _PEPSI_970_ID, "delta": -1})
    assert r.status_code == 200
    assert r.json()["cart"]["lines"] == []


def test_unknown_product_id_is_rejected_never_a_fabricated_line():
    session_id = "cart-endpoint-unknown"
    _reset(session_id)
    r = client.post(f"/session/{session_id}/cart/item", json={"product_id": 999_999_999, "delta": 1})
    assert r.status_code == 404
    assert get_activity_store().get(session_id).cart == []


def test_button_mutation_invalidates_an_open_checkout_draft():
    session_id = "cart-endpoint-invalidate"
    _reset(session_id)
    store = get_activity_store()

    product = DisplayedProduct(id=_PEPSI_970_ID, name_en="Pepsi - 970 Ml", name_ar="بيبسي - 970 مل",
                                brand_normalized="pepsi", price=21.75, special_price=21.75)
    store.add_to_cart(session_id, [product], quantity=1)
    store.set_selected_products(session_id, [CartLine(product=product, quantity=1)])
    store.start_checkout(session_id, cart_ids=[product.id])
    assert store.get(session_id).checkout_active is True

    r = client.post(f"/session/{session_id}/cart/item", json={"product_id": _PEPSI_970_ID, "delta": 1})
    assert r.status_code == 200
    assert r.json()["checkout_invalidated"] is True
    assert store.get(session_id).checkout_active is False
