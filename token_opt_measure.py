import logging
import sys
sys.path.insert(0, ".")

from src.scenario.orchestrator import handle_message
from src.scenario.session_activity import SessionActivityStore
from src.search_adapter.adapter import SearchAdapter
from src.state_manager.state_manager import StateManager
from src.understanding.llm import understand as real_understand

traces = []
last_action = {}


class _Capture(logging.Handler):
    def emit(self, record):
        if record.msg == "request_trace %s":
            traces.append(record.args)


logging.getLogger("src.scenario.orchestrator").addHandler(_Capture())
logging.getLogger("src.scenario.orchestrator").setLevel(logging.INFO)

results = []


def _tracking_understand(message, history=None):
    action = real_understand(message, history=history)
    last_action["value"] = action
    return action


def _search(f, l, o):
    return SearchAdapter().search(f, limit=l, offset=o)


def report(label, session_id, message, sm, store):
    before = len(traces)
    r = handle_message(session_id, message, state_manager=sm, activity_store=store,
                        understand_fn=_tracking_understand, search_fn=_search)
    trace = traces[before] if len(traces) > before else None
    action = last_action.get("value")
    results.append((label, message, action, r, trace))


sm1, store1 = StateManager(), SessionActivityStore()
report("A", "sA", "get me pepsi", sm1, store1)

sm2, store2 = StateManager(), SessionActivityStore()
report("B", "sB", "recommend a bag of chips", sm2, store2)

sm3, store3 = StateManager(), SessionActivityStore()
handle_message("sC", "show me pepsi", state_manager=sm3, activity_store=store3,
                understand_fn=_tracking_understand, search_fn=_search)
report("C", "sC", "add the first one to cart", sm3, store3)

sm4, store4 = StateManager(), SessionActivityStore()
handle_message("sD", "عايز بيبسي", state_manager=sm4, activity_store=store4,
                understand_fn=_tracking_understand, search_fn=_search)
report("D", "sD", "ضيف الأول للسلة", sm4, store4)

sm5, store5 = StateManager(), SessionActivityStore()
report("E", "sE", "now i want to buy", sm5, store5)

sm6, store6 = StateManager(), SessionActivityStore()
handle_message("sF", "show me pepsi", state_manager=sm6, activity_store=store6,
                understand_fn=_tracking_understand, search_fn=_search)
handle_message("sF", "add the first one to cart", state_manager=sm6, activity_store=store6,
                understand_fn=_tracking_understand, search_fn=_search)
handle_message("sF", "add the second one to cart", state_manager=sm6, activity_store=store6,
                understand_fn=_tracking_understand, search_fn=_search)
handle_message("sF", "checkout", state_manager=sm6, activity_store=store6,
                understand_fn=_tracking_understand, search_fn=_search)
report("F", "sF", "all of them", sm6, store6)

out_lines = []
for label, message, action, r, trace in results:
    out_lines.append("=" * 70)
    out_lines.append(f"[{label}] message={message!r}")
    out_lines.append(f"parsed_action={action!r}")
    out_lines.append(f"route={trace.get('route') if trace else None}")
    out_lines.append(f"llm_calls={trace.get('llm_calls') if trace else None}")
    out_lines.append(f"llm_input_tokens={trace.get('llm_input_tokens') if trace else None}")
    out_lines.append(f"llm_output_tokens={trace.get('llm_output_tokens') if trace else None}")
    out_lines.append(f"llm_total_tokens={trace.get('llm_total_tokens') if trace else None}")
    out_lines.append(f"checkout_form_present={r.checkout_form is not None}")
    out_lines.append(f"cart={r.cart}")
    out_lines.append(f"assistant_message={r.assistant_message!r}")

with open("token_opt_results.log", "w", encoding="utf-8") as f:
    f.write("\n".join(out_lines))

print("DONE")
