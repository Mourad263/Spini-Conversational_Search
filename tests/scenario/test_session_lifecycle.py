"""Sprint 6: session memory lifecycle (reset vs. expiration). Hermetic --
understand() always stubbed, same pattern as test_orchestrator.py. No real
OpenAI/Groq calls; `now` is the same injectable monotonic-clock float
handle_message already threads through to the rapid-repeat guard, reused
here to simulate inactivity without sleeping.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.scenario.orchestrator import handle_message  # noqa: E402
from src.scenario.session_activity import SESSION_TTL_SECONDS, SessionActivityStore  # noqa: E402
from src.search_adapter.adapter import FilterSet, SearchAdapter, SearchResult  # noqa: E402
from src.state_manager.state_manager import StateManager  # noqa: E402
from src.understanding.schema import Action, Intent  # noqa: E402

PAST_TTL = SESSION_TTL_SECONDS + 100.0


def stub_understand(*actions: Action):
    it = iter(actions)

    def _stub(message: str, history: list[str] | None = None) -> Action:
        return next(it)

    return _stub


class RecordingSearch:
    def __init__(self) -> None:
        self.calls: list[FilterSet] = []

    def __call__(self, filters: FilterSet, limit: int, offset: int) -> SearchResult:
        self.calls.append(filters)
        return SearchAdapter().search(filters, limit=limit, offset=offset)


# --- A: memory survives a normal shopping reset ---

def test_memory_survives_normal_search_reset():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"),
        Action(intent=Intent.RESET),
        Action(intent=Intent.NO_OP, memory_recall="first"),
    )
    search = RecordingSearch()
    handle_message("s1", "I need milk", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search, now=0.0)
    handle_message("s1", "start over", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search, now=1.0)

    r = handle_message("s1", "what was the first thing I asked?", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search, now=2.0)

    assert "I need milk" in r.assistant_message


# --- B: active shopping state is still cleared by reset ---

def test_reset_still_clears_active_shopping_state():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk", price_max=50.0),
        Action(intent=Intent.RESET),
    )
    search = RecordingSearch()
    handle_message("s1", "milk under 50", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search, now=0.0)

    r = handle_message("s1", "start over", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search, now=1.0)

    assert r.resolved_category is None
    assert r.price_max is None
    assert sm.get_state("s1") == FilterSet()


# --- C: an expired session loses conversation memory ---

def test_expired_session_loses_conversation_memory():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"),
        Action(intent=Intent.NO_OP, memory_recall="first"),
    )
    search = RecordingSearch()
    handle_message("s1", "I need milk", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search, now=0.0)

    r = handle_message("s1", "what was the first thing I asked?", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search, now=PAST_TTL)

    assert "I need milk" not in r.assistant_message
    assert "don't have" in r.assistant_message  # honest fallback, not a guess
    # the recall question itself is the ONLY thing in history now -- the old
    # "I need milk" turn is gone, confirming the expiry wipe actually happened.
    assert store.get("s1", now=PAST_TTL).history == ["what was the first thing I asked?"]


# --- D: an expired session loses shopping state too ---

def test_expired_session_loses_shopping_state():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk", price_max=50.0),
        Action(intent=Intent.NO_OP, recommendation_request=True),
    )
    search = RecordingSearch()
    handle_message("s1", "milk under 50", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search, now=0.0)
    assert sm.get_state("s1").category == "milk_29"

    r = handle_message("s1", "what do you recommend?", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search, now=PAST_TTL)

    assert r.resolved_category is None
    assert r.price_max is None
    assert sm.get_state("s1") == FilterSet()


# --- E: session isolation still holds under the TTL-aware store ---

def test_session_a_never_sees_session_b_memory():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"),
        Action(intent=Intent.NO_OP, memory_recall="previous"),
    )
    search = RecordingSearch()
    handle_message("session-a", "I need milk", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search, now=0.0)

    r = handle_message("session-b", "what did I say before?", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search, now=1.0)

    assert "milk" not in r.assistant_message.lower()
    assert sm.get_state("session-b") == FilterSet()


# --- F: a session well within the TTL still recalls prior turns normally ---

def test_non_expired_session_still_recalls_prior_turns():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"),
        Action(intent=Intent.NO_OP, memory_recall="previous"),
    )
    search = RecordingSearch()
    handle_message("s1", "I need milk", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search, now=0.0)

    r = handle_message("s1", "what did I say before?", state_manager=sm, activity_store=store,
                        understand_fn=understand_fn, search_fn=search, now=SESSION_TTL_SECONDS - 100.0)

    assert "I need milk" in r.assistant_message
    assert sm.get_state("s1").category == "milk_29"  # shopping state also survived


# ======================================================================
# Abandoned-session reclaim: get()'s own lazy expiry only ever revisits
# the SAME session_id being accessed -- an id that's simply never
# accessed again (a "New Chat" moving on to a fresh session_id) never
# gets cleaned up on its own. reclaim_abandoned() fixes that, amortized
# (SessionActivityStore._RECLAIM_CHECK_EVERY) rather than scanning on
# every request -- these tests drive the internal access counter directly
# to the amortization's trigger point (call #1) rather than looping
# _RECLAIM_CHECK_EVERY times, the same "reach into internal state"
# convention already used elsewhere in this suite for deterministic setup.
# ======================================================================

# --- G: an abandoned, expired session is reclaimed by a DIFFERENT
# session's access -- its own id is never revisited ---

def test_abandoned_expired_session_reclaimed_via_different_session_access():
    store = SessionActivityStore()
    store.get("old-session", now=0.0)  # abandoned -- never touched again

    store._access_count = 1  # force the amortized trigger point deterministically
    reclaimed = store.reclaim_abandoned(PAST_TTL, exclude="new-session")

    assert reclaimed == ["old-session"]
    assert "old-session" not in store._sessions


# --- H: its StateManager state is removed too (full orchestrator wiring) ---

def test_abandoned_session_state_manager_entry_also_removed():
    sm, store = StateManager(), SessionActivityStore()
    understand_fn = stub_understand(
        Action(intent=Intent.SEARCH, raw_query_text="milk", category_hint="milk"),
        Action(intent=Intent.SEARCH, raw_query_text="pepsi", brand_text="pepsi"),
    )
    search = RecordingSearch()
    handle_message("old-session", "I need milk", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search, now=0.0)
    assert sm.get_state("old-session").category == "milk_29"

    store._access_count = 0  # so the NEXT get() (inside handle_message below) lands on
    # access #1 -- the amortized reclaim trigger -- deterministically, not by chance
    handle_message("new-session", "I need pepsi", state_manager=sm, activity_store=store,
                    understand_fn=understand_fn, search_fn=search, now=PAST_TTL)

    assert "old-session" not in store._sessions
    assert sm.get_state("old-session") == FilterSet()  # StateManager's half reclaimed too


# --- I: a non-expired session survives the same reclaim sweep ---

def test_non_expired_session_survives_reclaim_sweep():
    store = SessionActivityStore()
    store.get("recent-session", now=0.0)

    store._access_count = 1
    reclaimed = store.reclaim_abandoned(SESSION_TTL_SECONDS - 100.0, exclude="new-session")

    assert reclaimed == []
    assert "recent-session" in store._sessions


# --- J: the CURRENT active session is never reclaimed, even if its own
# timestamp would otherwise look expired by the time of the sweep ---

def test_current_session_never_reclaims_itself():
    store = SessionActivityStore()
    store.get("s1", now=0.0)

    store._access_count = 1
    reclaimed = store.reclaim_abandoned(PAST_TTL, exclude="s1")

    assert reclaimed == []
    assert "s1" in store._sessions


# --- K: normal per-id expiry-on-access behavior (existing tests C/D
# above) is unaffected by the reclaim addition -- confirmed by those
# tests still passing unchanged in this same file.
