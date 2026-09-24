"""Sprint 6 recommendation-offer task: deterministic accept/reject
classification for a message following a pending recommendation offer
(src/scenario/session_activity.py's PendingAction). Same style/rigor as
src/scenario/social.py -- a tiny, bounded, EXACT whole-message match, never
a substring/keyword check. "yes, show me cheese" normalizes to something
NOT in either set and correctly falls through to the normal semantic
workflow rather than being swallowed as a bare acceptance. Deliberately a
separate module from social.py: these are two different vocabularies for
two different questions (is this message social chatter? vs. does this
message answer the specific yes/no Spini just asked?), and conflating them
would risk one set's forms leaking into the other's meaning.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ingestion.text_normalize import is_arabic, normalize_arabic, normalize_latin  # noqa: E402

ACCEPT = "accept"
REJECT = "reject"

_TERMINAL_PUNCT = re.compile(r"^[\s!?.,؟،؛:~*]+|[\s!?.,؟،؛:~*]+$")

_ACCEPT_EN = {"yes", "sure", "ok", "okay"}
_ACCEPT_AR = {"ايوه", "أيوه", "نعم", "تمام"}

_REJECT_EN = {"no", "no thanks", "nope", "cancel"}  # "cancel" added (cart/
# direct-checkout architecture task correction): a generic dismissal, not
# cart-specific -- equally valid for declining a recommendation offer or a
# multi-add-to-cart proposal, so it belongs in this shared bounded set.
_REJECT_AR = {"لا", "لأ", "مش عايز", "مش عايزة", "مش دلوقتي"}

_ACCEPT_FORMS = _ACCEPT_EN | _ACCEPT_AR
_REJECT_FORMS = _REJECT_EN | _REJECT_AR


def _clean(message: str) -> str:
    stripped = _TERMINAL_PUNCT.sub("", message)
    return normalize_arabic(stripped) if is_arabic(stripped) else normalize_latin(stripped)


def classify_offer_response(message: str) -> str | None:
    """ACCEPT/REJECT only when the message, in its ENTIRETY, matches one
    of the bounded known forms; None otherwise (including empty input) --
    callers must fall through to the normal workflow on None, never guess."""
    if not message or not message.strip():
        return None
    cleaned = _clean(message)
    if not cleaned:
        return None
    if cleaned in _ACCEPT_FORMS:
        return ACCEPT
    if cleaned in _REJECT_FORMS:
        return REJECT
    return None
