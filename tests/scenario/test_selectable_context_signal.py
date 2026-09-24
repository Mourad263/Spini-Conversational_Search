"""Sprint 6 product-selection follow-up: orchestrator.py must tell the real
understand() whether products are currently selectable, via
src/understanding/llm.py's contextvar side-channel (set_selectable_products_
shown/get_selectable_products_shown) -- never a new understand_fn parameter,
so every existing stubbed understand_fn keeps working unchanged. Hermetic:
the stub here simply reads the contextvar at call time to prove orchestrator
set it correctly, without needing a real Groq call.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.scenario.orchestrator import handle_message  # noqa: E402
from src.scenario.session_activity import SessionActivityStore  # noqa: E402
from src.search_adapter.adapter import FilterSet, SearchAdapter, SearchResult  # noqa: E402
from src.state_manager.state_manager import StateManager  # noqa: E402
from src.understanding.llm import get_selectable_products_shown  # noqa: E402
from src.understanding.schema import Action, Intent  # noqa: E402


class RecordingSearch:
    def __init__(self) -> None:
        self.calls: list[FilterSet] = []

    def __call__(self, filters: FilterSet, limit: int, offset: int) -> SearchResult:
        self.calls.append(filters)
        return SearchAdapter().search(filters, limit=limit, offset=offset)


def spying_understand(*actions: Action):
    """Records what get_selectable_products_shown() returned at the exact
    moment understand_fn was called, then returns the next queued Action."""
    it = iter(actions)
    seen: list[bool] = []

    def _stub(message: str, history: list[str] | None = None) -> Action:
        seen.append(get_selectable_products_shown())
        return next(it)

    return _stub, seen


def test_signal_is_false_before_anything_is_shown():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn, seen = spying_understand(Action(intent=Intent.OFF_TOPIC))
    search = RecordingSearch()

    handle_message("s1", "what's the weather?", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)

    assert seen == [False]


def test_signal_is_true_after_a_search_shows_products():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn, seen = spying_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"),
        Action(intent=Intent.OFF_TOPIC),
    )
    search = RecordingSearch()

    handle_message("s1", "show me milk", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    assert seen == [False]  # nothing shown yet for the FIRST turn

    handle_message("s1", "what's the weather?", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search)
    assert seen == [False, True]  # the milk search's real products are now selectable


def test_no_second_llm_call_introduced():
    """The signal is plumbed via a contextvar read inside the ONE existing
    understand() call -- confirms understand_fn is still invoked exactly
    once per turn, never twice."""
    sm, store = StateManager(), SessionActivityStore()
    call_count = 0

    def counting_understand(message, history=None):
        nonlocal call_count
        call_count += 1
        return Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk")

    search = RecordingSearch()
    handle_message("s1", "show me milk", state_manager=sm, activity_store=store,
                    understand_fn=counting_understand, search_fn=search)
    assert call_count == 1
