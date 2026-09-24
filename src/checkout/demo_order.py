"""Sprint 6 checkout task: DEMO order field validation + JSON persistence.

Deliberately no database table, no ORM: every successfully submitted demo
order is saved as its OWN JSON file under ORDERS_DIR, validated against the
one shared schemas/demo_order.schema.json BEFORE it is ever written.
"DEMO" is not decorative -- saving this JSON file means checkout succeeded
inside Project 6 only; it never means a real Spinneys order was received,
paid for, or dispatched (src/response/messages.py's confirmation wording
is equally careful about this).

Pure/session-independent: this module knows nothing about
SessionActivity/StateManager -- see src/scenario/checkout_submission.py for
the session-aware orchestration that calls into this.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jsonschema  # noqa: E402

from src.pricing import effective_price  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORDERS_DIR = _PROJECT_ROOT / "data" / "orders"
SCHEMA_PATH = _PROJECT_ROOT / "schemas" / "demo_order.schema.json"

# Egyptian mobile numbers: 01 + one of {0,1,2,5} + 8 more digits
# (Vodafone/Etisalat/Orange/WE prefixes), where the leading "0" is either
# kept (local form) or replaced by "+20" (international form) -- never
# both (a "0" after "+20") and never neither (a bare "1..." with no
# leading 0 at all). No carrier lookup, no external validation service,
# just a realistic shape check.
_PHONE_RE = re.compile(r"^(?:\+20|0)1[0125]\d{8}$")
_MIN_ADDRESS_LENGTH = 6  # rejects blank/trivially-short input without being overly strict


class DemoOrderValidationError(Exception):
    """A submitted field, or the built order itself, failed validation.
    Never raised after a file has been written -- see save_demo_order()."""


@dataclass
class ProductSnapshot:
    """The minimal product identity save_demo_order()/build_demo_order()
    need -- matches src/scenario/session_activity.py's DisplayedProduct
    shape so callers can pass that object directly, without this module
    importing session_activity (keeping it session-independent)."""

    id: int
    name_en: str
    name_ar: str
    price: float | None
    special_price: float | None


def validate_quantity(quantity: object) -> int:
    if isinstance(quantity, bool) or not isinstance(quantity, int):
        raise DemoOrderValidationError("quantity must be a whole number.")
    if quantity < 1:
        raise DemoOrderValidationError("quantity must be at least 1.")
    return quantity


def validate_customer_name(name: str) -> str:
    cleaned = (name or "").strip()
    if not cleaned:
        raise DemoOrderValidationError("customer_name is required.")
    return cleaned


def validate_phone(phone: str) -> str:
    cleaned = (phone or "").strip()
    digits_only = re.sub(r"[\s\-()]", "", cleaned)
    if not _PHONE_RE.match(digits_only):
        raise DemoOrderValidationError("phone does not look like a valid Egyptian mobile number.")
    return cleaned


def validate_delivery_address(address: str) -> str:
    cleaned = (address or "").strip()
    if len(cleaned) < _MIN_ADDRESS_LENGTH:
        raise DemoOrderValidationError("delivery_address is too short.")
    return cleaned


def validate_lines(lines: object, known_ids: set[int]) -> list[tuple[int, int]]:
    """Sprint 6 multi-product checkout: `lines` must be a non-empty list of
    (product_id, quantity), each product_id from `known_ids` (the active
    checkout's own server-known selection) exactly once -- never a subset
    (silently dropping a line), never an id the session never selected.
    Returns [(product_id, quantity), ...] in the given order."""
    if not lines:
        raise DemoOrderValidationError("at least one checkout line is required.")
    seen: set[int] = set()
    result: list[tuple[int, int]] = []
    for line in lines:
        pid, qty = line
        if pid in seen:
            raise DemoOrderValidationError(f"duplicate checkout line for product {pid}.")
        seen.add(pid)
        result.append((pid, validate_quantity(qty)))
    if seen != known_ids:
        raise DemoOrderValidationError("submitted lines do not match the active checkout selection.")
    return result


def _unit_price(snapshot: ProductSnapshot) -> Decimal:
    raw = effective_price(snapshot.price, snapshot.special_price)
    if raw is None:
        raise DemoOrderValidationError("selected product has no known price.")
    return Decimal(str(raw))


def build_demo_order(
    session_id: str, lines: list[tuple[ProductSnapshot, int]],
    customer_name: str, phone: str, delivery_address: str,
) -> dict:
    """Builds the server-authoritative DemoOrder dict -- product identity
    and price for EVERY line come ONLY from its own `snapshot` (the
    session's own selected-product records), never from any client-
    submitted value. `lines` is (snapshot, quantity) pairs, already
    de-duplicated/matched against the session's known selection by
    validate_lines(). Caller must already have validated quantity/
    customer_name/phone/delivery_address. Raises on the FIRST unpriced
    line -- an unknown-price product never becomes a free/zero line, and
    no partial order (some lines priced, one silently dropped) is ever
    built."""
    order_lines = []
    total_price = Decimal("0")
    for snapshot, quantity in lines:
        unit_price = _unit_price(snapshot)
        line_total = (unit_price * quantity).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        total_price += line_total
        order_lines.append({
            "id": snapshot.id,
            "name_en": snapshot.name_en,
            "name_ar": snapshot.name_ar,
            "quantity": quantity,
            "unit_price": float(unit_price),
            "line_total": float(line_total),
        })
    return {
        "$schema": "../../schemas/demo_order.schema.json",
        "order_id": uuid.uuid4().hex,
        "session_id": session_id,
        "products": order_lines,
        "total_price": float(total_price),
        "customer": {
            "name": customer_name,
            "phone": phone,
            "delivery_address": delivery_address,
        },
        "payment_method": "cash_on_delivery",
        "status": "submitted",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def save_demo_order(
    order: dict, *, orders_dir: Path | None = None, schema_path: Path | None = None,
) -> Path:
    """Validates `order` against schemas/demo_order.schema.json, then
    writes it atomically (temp file in the SAME directory + os.replace,
    atomic on both POSIX and Windows) as its own <order_id>.json. Raises
    DemoOrderValidationError/jsonschema.ValidationError/OSError and leaves
    NO final file behind on any failure -- a partially written temp file is
    cleaned up, never left as (or renamed to) the final path."""
    orders_dir = orders_dir or ORDERS_DIR
    schema_path = schema_path or SCHEMA_PATH

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.validate(instance=order, schema=schema)  # raises jsonschema.ValidationError

    order_id = order["order_id"]
    # order_id is always our OWN uuid4().hex (never derived from customer
    # name/phone/address/product name -- see build_demo_order()), but this
    # stays defensive against path traversal regardless of caller.
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", order_id or ""):
        raise DemoOrderValidationError("invalid order id.")

    orders_dir.mkdir(parents=True, exist_ok=True)
    final_path = orders_dir / f"{order_id}.json"
    if final_path.exists():
        # Never overwrite an existing order -- a real uuid4 collision is
        # astronomically unlikely; treat it as a hard failure, not a retry,
        # since retrying here would need a NEW order_id (a new build_demo_order()
        # call), which is the caller's decision, not this function's.
        raise DemoOrderValidationError("order id collision -- refusing to overwrite an existing order.")

    fd, tmp_path_str = tempfile.mkstemp(prefix=f".{order_id}-", suffix=".tmp", dir=orders_dir)
    tmp_path = Path(tmp_path_str)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(order, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, final_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return final_path
