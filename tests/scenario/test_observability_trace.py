"""Instructor-demo observability task: SPINI_TRACE=1 prints one compact,
human-readable execution trace per request (src/scenario/trace.py's
format_human_trace(), populated by src/scenario/orchestrator.py's
handle_message()/_handle_message_inner() and src/scenario/
checkout_submission.py's submit_checkout()). Purely observational -- never
influences routing/behavior/API response. These tests only check the
printed trace itself, never re-test search/cart/checkout behavior (see the
existing scenario test files for that).

Hermetic -- `understand_fn` is always stubbed; no real LLM/DB calls.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.scenario.orchestrator import handle_message  # noqa: E402
from src.scenario.session_activity import DisplayedProduct, SessionActivityStore  # noqa: E402
from src.search_adapter.adapter import FilterSet, SearchResult  # noqa: E402
from src.state_manager.state_manager import StateManager  # noqa: E402
from src.understanding.schema import Action, Intent  # noqa: E402


def stub_understand(*actions: Action):
    it = iter(actions)

    def _stub(message: str, history: list[str] | None = None) -> Action:
        return next(it)

    return _stub


class StubSearch:
    """No real DB/SearchAdapter -- returns one fixed product."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, filters: FilterSet, limit: int, offset: int) -> SearchResult:
        self.calls += 1
        return SearchResult(
            products=[{
                "id": 1, "name_en": "Pepsi - 970 Ml", "name_ar": "بيبسي - 970 مل",
                "price": 21.75, "special_price": 21.75, "brand_normalized": "pepsi",
                "url_key": "pepsi-970ml", "sku": "5293",
            }],
            total_count=1,
            zero_result=False,
        )


_PRODUCT = DisplayedProduct(id=1, name_en="Pepsi - 970 Ml", name_ar="بيبسي - 970 مل",
                             brand_normalized="pepsi", price=21.75, special_price=21.75)


# --- A: SPINI_TRACE=1 prints the full block; unset prints nothing ---

def test_trace_printed_only_when_env_toggle_enabled(monkeypatch, capsys):
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(Action(intent=Intent.SEARCH, raw_query_text="pepsi", brand_text="pepsi"))
    search = StubSearch()

    monkeypatch.delenv("SPINI_TRACE", raising=False)
    handle_message("s1", "show me pepsi", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    assert "[SPINI TRACE]" not in capsys.readouterr().out

    monkeypatch.setenv("SPINI_TRACE", "1")
    handle_message("s1", "show me pepsi again", state_manager=sm, activity_store=store,
                    understand_fn=stub_understand(Action(intent=Intent.SEARCH, raw_query_text="pepsi", brand_text="pepsi")),
                    search_fn=search)
    out = capsys.readouterr().out
    assert "[SPINI TRACE]" in out
    for section in ("MESSAGE", "STATE BEFORE", "INTERPRETATION", "RESOLUTION",
                    "ROUTE", "STATE MUTATION", "STATE AFTER", "RESPONSE"):
        assert section in out


# --- B: fast path (no LLM call) vs a real understand() call are
# distinguishable in the printed trace ---

def test_trace_distinguishes_fast_path_from_llm_called(monkeypatch, capsys):
    sm, store = StateManager(), SessionActivityStore()
    monkeypatch.setenv("SPINI_TRACE", "1")

    handle_message("s1", "hello", state_manager=sm, activity_store=store,
                    understand_fn=stub_understand(), search_fn=StubSearch())
    fast_path_out = capsys.readouterr().out
    assert "fast_path=True" in fast_path_out
    assert "llm_called=False" in fast_path_out

    handle_message("s1", "show me pepsi", state_manager=sm, activity_store=store,
                    understand_fn=stub_understand(Action(intent=Intent.SEARCH, raw_query_text="pepsi", brand_text="pepsi")),
                    search_fn=StubSearch())
    llm_out = capsys.readouterr().out
    assert "fast_path=False" in llm_out
    assert "llm_called=True" in llm_out


# --- C: cart add by exact title -- resolution authority/product_ids show
# in the trace ---

def test_trace_shows_resolution_authority_for_cart_add(monkeypatch, capsys):
    sm, store = StateManager(), SessionActivityStore()
    store.set_last_shown_products("s1", [_PRODUCT])
    monkeypatch.setenv("SPINI_TRACE", "1")

    handle_message("s1", "add Pepsi - 970 Ml to cart", state_manager=sm, activity_store=store,
                    understand_fn=stub_understand(Action(intent=Intent.NO_OP, cart_action="add")),
                    search_fn=StubSearch())
    out = capsys.readouterr().out
    assert "authority=last_shown_products+recent_shown" in out
    assert "product_ids=[1]" in out
    assert "reason=ok" in out
    assert "cart: 0 -> 1" in out


# --- D: checkout submission never logs PII, only metadata ---

def test_trace_checkout_submission_logs_metadata_not_pii(monkeypatch, capsys, tmp_path):
    from src.scenario.checkout_submission import submit_checkout

    sm, store = StateManager(), SessionActivityStore()
    store.set_last_shown_products("s1", [_PRODUCT])
    store.append_history("s1", "show me pepsi", "search")
    store.set_selected_products("s1", [__import__("src.scenario.session_activity", fromlist=["CartLine"]).CartLine(product=_PRODUCT, quantity=1)])
    store.start_checkout("s1")

    monkeypatch.setenv("SPINI_TRACE", "1")
    submit_checkout(
        "s1", lines=[(1, 1)], customer_name="Ahmed Mohamed", phone="+201001234567",
        delivery_address="123 Some Real Street, Cairo", state_manager=sm, activity_store=store,
        orders_dir=tmp_path / "orders",
    )
    out = capsys.readouterr().out
    assert "Ahmed Mohamed" not in out
    assert "+201001234567" not in out
    assert "Some Real Street" not in out
    assert "name_present=True" in out
    assert "phone_valid=True" in out
    assert "address_present=True" in out
    assert "items=1" in out
