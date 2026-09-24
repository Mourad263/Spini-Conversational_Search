"""Sprint 6 social fast-path: recognizes ONLY clearly social-only turns
(greeting / how-are-you / gratitude) so src/scenario/orchestrator.py can
answer them deterministically before understand() is ever called -- zero
LLM/validation/search/recommendation/embedding calls for these.

Deliberately a tiny, bounded set of normalized whole-message forms, not a
phrase dictionary or a semantic classifier: classify_social() only ever
matches the message in its ENTIRETY (after stripping terminal punctuation,
ordinary apostrophes -- a contraction is a surface/punctuation variant, not
a spelling mistake, so "what's up"/"whats up" become the SAME normalized
string here rather than relying on fuzzy tolerance for it -- and the same
normalize_arabic()/normalize_latin() the rest of the search engine already
uses) -- exactly, or via a strict rapidfuzz ratio() against one of the same
small canonical forms (a minor spelling slip like "hallo"/"hello", never a
substring/partial match, which is why this uses ratio() and not the
catalog's WRatio() -- WRatio's partial-ratio component scores "hello show
me milk" ~90 against "hello" alone, exactly the false-positive whole-
message safety must prevent). The fuzzy fallback also requires the same
FIRST character as the canonical form: real misspellings overwhelmingly
preserve a word's initial letter (substitution/omission/transposition
happens later in the word), while an unrelated same-length real word
("cello"/"jello" against "hello") that coincidentally sits at the same edit
distance almost never shares it -- this is what actually separates a
genuine typo from a same-distance coincidence, not a higher threshold
(raising the threshold would also reject "hallo" itself, which sits at the
SAME 80.0 ratio as "cello"). A message with any additional content --
"hello, I need milk", "thanks, show me more" -- is far below threshold
against every canonical form and correctly falls through to the real
understand() workflow. When uncertain, this always returns None; it must
never swallow a message that might carry shopping meaning.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rapidfuzz import fuzz  # noqa: E402

from src.ingestion.text_normalize import is_arabic, normalize_arabic, normalize_latin  # noqa: E402

GREETING = "greeting"
HOW_ARE_YOU = "how_are_you"
GRATITUDE = "gratitude"
NAME_QUESTION = "name_question"

_TERMINAL_PUNCT = re.compile(r"^[\s!?.,؟،؛:~*]+|[\s!?.,؟،؛:~*]+$")
_APOSTROPHE = re.compile(r"[’']")  # a contraction's apostrophe is a
# punctuation surface variant ("what's" / "whats"), not a spelling mistake
# -- normalized away here so fuzzy matching is never needed just to absorb it.

_FUZZY_THRESHOLD = 80  # same threshold convention as orchestrator.py's
# _SELECTION_TEXT_FUZZY_THRESHOLD -- a minor spelling slip against its real
# canonical form scores 80+ here; an unrelated message (different length/
# content) scores well below it.
_MIN_FUZZY_FORM_LEN = 5  # canonical forms shorter than this ("hi", "hey")
# are exact-match only -- too little room for ratio() to tell a real typo
# apart from an unrelated short word of similar length ("hit" ~ "hi").

_GREETINGS_EN = {"hi", "hello", "hey", "hiya", "good morning", "good evening", "good afternoon"}
_GREETINGS_AR = {"مرحبا", "أهلا", "اهلا", "السلام عليكم", "صباح الخير", "مساء الخير"}

_HOW_ARE_YOU_EN = {"how are you", "how are you doing", "hows it going", "whats up"}
_HOW_ARE_YOU_AR = {"ازيك", "ازيكي", "ازيكم", "عامل ايه", "عاملة ايه", "اخبارك ايه", "ايه اخبارك"}

_GRATITUDE_EN = {"thanks", "thank you", "thanks a lot", "thank you so much", "many thanks", "appreciate it"}
_GRATITUDE_AR = {"شكرا", "متشكر", "متشكرة", "تسلم", "تسلمي", "مشكور", "مشكورة"}

_NAME_QUESTION_EN = {"whats your name", "what is your name", "who are you"}
_NAME_QUESTION_AR = {"اسمك ايه", "اسمك إيه", "ما اسمك", "مين انت", "مين أنت"}

_CATEGORY_FORMS: tuple[tuple[str, set[str]], ...] = (
    (GREETING, _GREETINGS_EN | _GREETINGS_AR),
    (HOW_ARE_YOU, _HOW_ARE_YOU_EN | _HOW_ARE_YOU_AR),
    (GRATITUDE, _GRATITUDE_EN | _GRATITUDE_AR),
    (NAME_QUESTION, _NAME_QUESTION_EN | _NAME_QUESTION_AR),
)


def _clean(message: str) -> str:
    stripped = _TERMINAL_PUNCT.sub("", message)
    normalized = normalize_arabic(stripped) if is_arabic(stripped) else normalize_latin(stripped)
    return _APOSTROPHE.sub("", normalized)


def classify_social(message: str) -> str | None:
    """GREETING/HOW_ARE_YOU/GRATITUDE if the message, in its ENTIRETY,
    matches (exactly, or as a close rapidfuzz variant) one of the bounded
    known social forms; None otherwise (including empty/whitespace-only
    input, which is never a real social turn -- callers already reject
    empty messages upstream, but this stays defensive)."""
    if not message or not message.strip():
        return None
    cleaned = _clean(message)
    if not cleaned:
        return None
    for label, forms in _CATEGORY_FORMS:
        if cleaned in forms:
            return label
        if any(
            len(form) >= _MIN_FUZZY_FORM_LEN
            and cleaned[:1] == form[:1]
            and fuzz.ratio(cleaned, form) >= _FUZZY_THRESHOLD
            for form in forms
        ):
            return label
    return None
