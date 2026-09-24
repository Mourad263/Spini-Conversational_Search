"""Orchestrator organization refactor: product-selection/reference-
resolution, moved verbatim out of src/scenario/orchestrator.py. Everything
here is deterministic -- exact raw-title resolution, named (fuzzy)
selection, ordinal/last handling, multi-item ("buy X and Y") resolution,
and the bounded recent-shown-context pool -- never a second, competing
matching engine, and never a catalog-wide search. `_resolve_product_
selection()` is the ONE resolver every caller (plain shown-product
selection, cart add/remove/update-quantity, direct checkout, cart-subset
checkout, pending cart-checkout clarification) uses; see its own docstring
for the full authority rules this module enforces.
"""

from __future__ import annotations

import re

from rapidfuzz import fuzz

from src.ingestion.text_normalize import is_arabic, normalize_arabic, normalize_latin
from src.response import messages
from src.scenario.session_activity import DisplayedProduct
from src.scenario.trace import RequestTrace
from src.understanding.schema import Action

_SELECTION_TEXT_FUZZY_THRESHOLD = 80  # same threshold convention as the fuzzy
# category/brand matching in keyword_baseline.py -- rapidfuzz's WRatio scores a
# genuine (possibly misspelled) reference to a shown product's name/brand at
# ~85+ while an unrelated candidate stays in the 20s/30s, so one fixed cutoff
# cleanly separates "real match" from "not this one" without a second,
# catalog-wide fuzzy system.


def _normalize_field(field: str) -> str:
    """The project's one Arabic/Latin normalization convention."""
    return normalize_arabic(field) if is_arabic(field) else normalize_latin(field)


_MULTI_ITEM_SPLIT_RE = re.compile(r"\s*(?:,|،|&)\s*|\s+and\s+|\s+و\s+|\s+و(?=\S)", re.IGNORECASE)
# bounded connector vocabulary (same "small bounded marker list" convention as
# _LAST_MARKERS/_ORDINAL_WORDS_EN below) -- candidate item-boundary markers
# ONLY -- a connector occurrence is never blindly treated as a real boundary
# (real catalog names legitimately contain these: "...Salt and Vinegar...",
# "Cheese & Onion...", "Shrimp, Medium..."); see _split_named_items(), which
# uses recent_shown itself to tell a real product-name connector apart from
# one that actually separates two requested items. The last alternative
# (correction) handles natural Egyptian Arabic ATTACHED conjunction --
# "البيبسي والسردين" has no space after و -- matched only at a token
# boundary (preceded by whitespace, immediately followed by a non-space
# character, i.e. و starting a new word) so it can never fire on a و that is
# merely a letter INSIDE an earlier word (never preceded by whitespace
# there). The already-spaced form above still matches its own case first
# ("و شيء") since this attached form requires NO space right after و.
# one that actually separates two requested items.


def _maximal_title_matches(candidates: list[DisplayedProduct]) -> list[DisplayedProduct]:
    """Cross-turn multi-product checkout task ("IMPORTANT MATCHING SAFETY"
    correction): drops any candidate whose own shown name is a substring of
    ANOTHER candidate's shown name in this same list -- a shorter real
    product name that only "matched" because it happens to be contained
    inside a longer, DIFFERENT shown product's name (e.g. "Nagel Fish - Kg"
    literally inside "Nagel Fish - Kg Family Pack") must never turn one
    real mention into two checkout lines, or a false ambiguity. Keeps only
    the maximal (most specific) matches."""
    def _nested_in_another(candidate: DisplayedProduct) -> bool:
        for other in candidates:
            if other.id == candidate.id:
                continue
            if (candidate.name_en and other.name_en
                    and _normalize_field(candidate.name_en) in _normalize_field(other.name_en)):
                return True
            if (candidate.name_ar and other.name_ar
                    and _normalize_field(candidate.name_ar) in _normalize_field(other.name_ar)):
                return True
        return False
    return [c for c in candidates if not _nested_in_another(c)]


def _exact_title_matches(text: str, pool: list[DisplayedProduct]) -> list[DisplayedProduct]:
    """Shown products from `pool` whose complete normalized name is a
    literal substring of `text`, maximal-filtered (see
    _maximal_title_matches) so a shorter name nested inside a longer
    matched name is never double-counted."""
    latin = normalize_latin(text)
    arabic = normalize_arabic(text)
    hits = [
        p for p in pool
        if (p.name_en and _normalize_field(p.name_en) in latin)
        or (p.name_ar and _normalize_field(p.name_ar) in arabic)
    ]
    return _maximal_title_matches(hits) if len(hits) > 1 else hits


def _protected_spans(raw_message: str, recent_shown: list[DisplayedProduct]) -> list[tuple[int, int]]:
    """Product-aware segmentation (Blocker correction): character spans in
    `raw_message` covered by a recently shown product's OWN name appearing
    literally (case-insensitive for name_en) in the raw text. A connector
    falling inside one of these spans is part of a REAL product name, not
    an item separator -- recent_shown (the same bounded pool matching
    already uses) is the known vocabulary that tells the two apart; no new
    matching engine, just a literal search for each candidate's own name."""
    spans: list[tuple[int, int]] = []
    for p in recent_shown:
        if p.name_en:
            spans.extend(m.span() for m in re.finditer(re.escape(p.name_en), raw_message, re.IGNORECASE))
        if p.name_ar and p.name_ar in raw_message:
            start = raw_message.index(p.name_ar)
            spans.append((start, start + len(p.name_ar)))
    return spans


def _split_named_items(raw_message: str, recent_shown: list[DisplayedProduct]) -> list[str]:
    """Product-aware segmentation (Blocker correction): splits on
    _MULTI_ITEM_SPLIT_RE's connectors EXCEPT where a connector occurrence
    falls inside a _protected_spans() span -- so "...Chips With Salt and
    Vinegar - 130G and Sardine Fish" splits into exactly the 2 real items
    (the "and" inside the shown product's own name is never a boundary),
    "Cheese & Onion..." and "Shrimp, Medium..." are protected the same way.
    Returns [raw_message] unsplit when there is no valid (unprotected)
    connector at all -- the caller then treats it as a single reference."""
    protected = _protected_spans(raw_message, recent_shown)
    cuts = [
        m.span() for m in _MULTI_ITEM_SPLIT_RE.finditer(raw_message)
        if not any(m.start() < p_end and m.end() > p_start for p_start, p_end in protected)
    ]
    if not cuts:
        return [raw_message]

    segments = []
    pos = 0
    for start, end in cuts:
        segments.append(raw_message[pos:start])
        pos = end
    segments.append(raw_message[pos:])
    return [s.strip() for s in segments if s.strip()]


_LAST_MARKERS = ("last", "اخر", "اخير")  # English "last"; Arabic "آخر"
# (normalize_arabic's alef-variant folding turns "آخر" into "اخر") and
# "الاخير" -- schema.py's own documented selection_last trigger forms.
_ORDINAL_WORDS_EN = (
    "first", "second", "third", "fourth", "fifth",
    "sixth", "seventh", "eighth", "ninth", "tenth",
)
_ORDINAL_WORDS_AR = {  # cart/direct-checkout architecture task (correction):
    # bounded Egyptian-colloquial + MSA ordinal vocabulary, positions 1-10 --
    # the same range _ORDINAL_WORDS_EN covers, for parity. Grammar, not
    # catalog truth -- one small fixed table, never a product-specific map.
    # normalize_arabic() already folds alef-hamza variants (أ/إ/آ -> ا), so
    # e.g. "أول" is matched via its folded form "اول" below; "ال"-prefixed
    # (definite article) forms don't get stripped by normalization, so both
    # bare and "ال"-prefixed spellings are listed explicitly where they differ.
    1: ("اول", "الاول"),
    2: ("تاني", "التاني", "ثاني", "الثاني"),
    3: ("تالت", "التالت", "ثالث", "الثالث"),
    4: ("رابع", "الرابع"),
    5: ("خامس", "الخامس"),
    6: ("سادس", "السادس"),
    7: ("سابع", "السابع"),
    8: ("تامن", "التامن", "ثامن", "الثامن"),
    9: ("تاسع", "التاسع"),
    10: ("عاشر", "العاشر"),
}
_ARABIC_INDIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def _raw_message_supports_last(raw_message: str) -> bool:
    """True only when the user's own words actually contain a "last"
    marker -- never inferred from an LLM field (see
    _resolve_product_selection())."""
    return any(marker in normalize_latin(raw_message) + normalize_arabic(raw_message) for marker in _LAST_MARKERS)


def _raw_message_supports_ordinal(raw_message: str, ordinal: int) -> bool:
    """True only when the user's own words actually contain THIS specific
    position -- a real numeric TOKEN (not a substring: "10"/"12" must never
    falsely support position "1" or "2" -- digit-adjacency lookaround, not
    `in`), ASCII or Arabic-Indic (١٢٣...), a small bounded English ordinal-
    word table, or the same small bounded Arabic ordinal table
    (_ORDINAL_WORDS_AR, positions 1-10) -- the ONE place raw positional
    evidence is checked, reused for every positional reference (single or
    multi-position, shown-product or cart) rather than a separate
    implementation per context."""
    ascii_digits = raw_message.translate(_ARABIC_INDIC_DIGITS)
    if re.search(rf"(?<!\d){ordinal}(?!\d)", ascii_digits):
        return True
    if 1 <= ordinal <= len(_ORDINAL_WORDS_EN) and _ORDINAL_WORDS_EN[ordinal - 1] in normalize_latin(raw_message):
        return True
    return any(word in normalize_arabic(raw_message) for word in _ORDINAL_WORDS_AR.get(ordinal, ()))


def _is_pure_positional_word(segment: str) -> bool:
    """True when `segment` (after connector-splitting) is a POSITIONAL
    reference -- a bare digit/ordinal/last marker on its own ("first"), or
    (correction) that same word wrapped in simple filler ("checkout
    first", "buy the third", "the last one") -- never a product-name
    candidate. See _positional_token() for how the actual digit/ordinal
    word is located within a possibly filler-wrapped segment -- the SAME
    extraction _positional_index() below reuses to compute the real index,
    never a separate check. Excludes such segments from
    _match_named_products()'s fuzzy fallback entirely -- a generic short
    word like "first"/"last" must never be fuzzy-matched against an
    unrelated product purely on rapidfuzz WRatio's own length-mismatch
    scoring quirk (real symptom: "first and last" scored both words 90+
    against "Sardine Fish" alone, the same quirk already documented on
    _match_segment_to_product())."""
    text = segment.strip()
    if not text:
        return True
    if _raw_message_supports_last(text):
        return True
    return _positional_token(text) is not None


def _selection_text_score(text_norm: str, product: DisplayedProduct) -> float:
    """Best rapidfuzz WRatio between the user's (already brief, LLM-
    extracted) selection_text and any of this shown product's own
    identifying fields. WRatio (unlike plain substring containment) tolerates
    the ordinary spelling variation a real conversation produces ("fa
    yoghart" for "Fa Yoghurt...") while still scoring an unrelated product
    far below threshold -- see _SELECTION_TEXT_FUZZY_THRESHOLD."""
    if not text_norm:
        return 0.0
    best = 0.0
    for field in (product.name_en, product.name_ar, product.brand_normalized or ""):
        if not field:
            continue
        best = max(best, fuzz.WRatio(text_norm, _normalize_field(field)))
    return best


def _match_segment_to_product(segment: str, pool: list[DisplayedProduct]) -> tuple[DisplayedProduct | None, str]:
    """Resolves ONE non-positional segment against `pool` via exact-title-
    then-fuzzy matching -- the ONE per-segment named-matching primitive,
    shared by _match_named_products() (pure named multi-item messages) and
    _resolve_product_selection()'s mixed named+positional handling (never
    duplicated). Returns (product, "ok") on a unique match, else
    (None, "ambiguous"/"not_found") -- same calibrated threshold/ranked-
    winner rule as everywhere else in this module."""
    exact = _exact_title_matches(segment, pool)
    if len(exact) == 1:
        return exact[0], "ok"
    if len(exact) > 1:
        return None, "ambiguous"

    # Fuzzy fallback: the SAME calibrated scoring/threshold as ordinary
    # single selection, but ranked rather than counted -- WRatio can give
    # several unrelated, differently-shaped candidates a score above the
    # bare threshold at once (real quirk: a short shown name can score 85+
    # via partial-ratio credit against a long, wholly unrelated segment
    # purely on length/character overlap), so requiring "everyone >=
    # threshold" would falsely flag a genuinely clear winner (e.g. a 95 vs.
    # two unrelated 85s) as ambiguous. A unique STRONG match means the top
    # score clearly stands alone; only a real tie at the top (no dominant
    # winner at all) is ambiguous.
    text_norm = _normalize_field(segment)
    ranked = sorted(pool, key=lambda p: _selection_text_score(text_norm, p), reverse=True)
    if not ranked:
        return None, "not_found"
    best, best_score = ranked[0], _selection_text_score(text_norm, ranked[0])
    if best_score < _SELECTION_TEXT_FUZZY_THRESHOLD:
        return None, "not_found"
    if len(ranked) > 1 and _selection_text_score(text_norm, ranked[1]) >= best_score:
        return None, "ambiguous"
    return best, "ok"


def _ordinal_value_of_word(word: str) -> int | None:
    """The 1-based position a positional ORDINAL WORD refers to (never a
    bare digit or a "last" marker -- callers handle those separately),
    reusing the SAME bounded vocabularies as raw positional evidence-
    checking (_ORDINAL_WORDS_EN/_ORDINAL_WORDS_AR) -- "third"/"third" -> 3,
    "التالت" -> 3. None when `word` isn't a recognized ordinal word."""
    latin = normalize_latin(word)
    if latin in _ORDINAL_WORDS_EN:
        return _ORDINAL_WORDS_EN.index(latin) + 1
    arabic = normalize_arabic(word)
    for ordinal, words in _ORDINAL_WORDS_AR.items():
        if arabic in words:
            return ordinal
    return None


def _positional_token(segment: str) -> str | None:
    """Returns the segment's own FIRST or LAST token if it is a bare digit
    or a recognized ordinal word, else None -- the ONE place a digit/
    ordinal token is located within a segment that may carry simple filler
    around it ("checkout first" -> "first", "buy the third" -> "third"),
    shared by _is_pure_positional_word() (just needs to know one exists)
    and _positional_index() (needs the actual token to compute the real
    position). Checking only the first/last token, never ones in the
    middle, avoids misreading an embedded quantity as positional (e.g.
    "buy 2 bottles of Pepsi" -- "2" is a middle token, "buy" and "Pepsi"
    are not digits/ordinal words, so this correctly returns None)."""
    words = [w for w in re.split(r"\s+", segment.strip()) if w]
    if not words:
        return None
    for candidate in (words[-1], words[0]):
        cleaned = candidate.strip(".,!?؟،؛:")
        if cleaned and (cleaned.isdigit() or _ordinal_value_of_word(cleaned) is not None):
            return cleaned
    return None


def _positional_index(
    segment: str, current_shown: list[DisplayedProduct], raw_message: str,
) -> tuple[int | None, str]:
    """Resolves ONE pure-positional segment (_is_pure_positional_word) to a
    0-based index into current_shown -- a bare digit/ordinal word located
    via _positional_token() (possibly wrapped in filler), or a "last"
    marker (same elimination/raw-evidence rule selection_last uses
    elsewhere). No separate raw-evidence check is needed for the digit/
    ordinal-word case: the token itself came FROM splitting the raw
    message, so its presence there is already proven. Returns
    (index, "ok"), or (None, "ambiguous"/"out_of_range")."""
    token = _positional_token(segment)
    if token is not None:
        ordinal = int(token) if token.isdigit() else _ordinal_value_of_word(token)
    else:
        # No digit/ordinal-word token found -- the only other way
        # _is_pure_positional_word() could have accepted this segment is a
        # recognized "last" marker.
        if len(current_shown) == 1 or _raw_message_supports_last(raw_message):
            return len(current_shown) - 1, "ok"
        return None, "ambiguous"
    idx = ordinal - 1
    if idx < 0 or idx >= len(current_shown):
        return None, "out_of_range"
    return idx, "ok"


def _match_named_products(
    raw_message: str, recent_shown: list[DisplayedProduct],
) -> tuple[list[DisplayedProduct], str] | None:
    """Cross-turn multi-product checkout task (Blocker correction): the ONE
    place the raw message is checked for explicitly NAMED recent products.
    Returns None when the message shows NO evidence of naming any recent
    product at all -- the caller then falls through to
    ordinal/last/selection_text (selection_text against the CURRENT page
    only, never this function's bounded `recent_shown` pool -- see
    _resolve_product_selection(); an ordinary single fuzzy reference must
    not risk interference from an older page). Otherwise returns
    (products, reason): "ok" ONLY when EVERY named item resolves to EXACTLY
    ONE product -- one unresolved or ambiguous item fails the WHOLE
    request, never a partial checkout of only SOME of what was named (real
    symptom: "buy Product A and Completely Unknown Product C" used to
    silently return just [A], "ok").

    A single reference (no VALID connector -- see _split_named_items(),
    which is product-aware: a connector inside a recently shown product's
    own name, e.g. "...Salt and Vinegar...", is never a real boundary) is
    exact-title-only here, exactly the pre-existing single/same-turn-
    multi-title behavior, just scoped to the bounded recent pool instead of
    only the current page: 0 exact hits means "no named evidence" (None,
    falls through), 1 hit resolves it, 2+ DISTINCT hits is the existing
    same-turn "buy X Y" (no connector) multi-select.

    Connector evidence (2+ segments AFTER product-aware splitting) is what
    unlocks the per-segment fuzzy fallback -- reusing _selection_text_score,
    no second matching engine -- since that evidence alone is what proves
    the user is naming more than one thing (real symptom: shown title
    "Nagel Fish - Kg" vs typed "Nagel Fish - 1 Kg" needs fuzzy tolerance,
    but ONLY when a real multi-item request is already established, never
    for an ordinary single mention). Each segment must resolve to exactly
    one product (exact title first, then fuzzy only if no exact hit) or the
    whole request fails outright."""
    segments = _split_named_items(raw_message, recent_shown)

    if len(segments) < 2:
        exact = _exact_title_matches(raw_message, recent_shown)
        return (exact, "ok") if exact else None

    # Multi-positional correction: a connector-split message is NOT
    # automatically a named-product attempt -- "first and third" splits
    # into ["first", "third"], neither of which is (or should be
    # fuzzy-matched against) a product name at all; that is a POSITIONAL
    # reference (selection_ordinals), handled entirely separately below in
    # _resolve_product_selection(). Real symptom this fixes: this function
    # used to always commit to a definitive named-reference verdict (ok/
    # ambiguous/not_found) the instant 2+ segments existed, which silently
    # swallowed every multi-position turn as a bogus "not_found" before the
    # ordinal branch was ever reached. The fix: track whether ANY segment
    # actually matched something real; only once that is true does a LATER
    # unmatched/ambiguous segment count as a genuine failed named request
    # (Blocker: "buy Product A and Completely Unknown Product C" must still
    # fail as incomplete, not silently defer to a positional read that
    # can't explain "Product A" either). Zero segments matching anything at
    # all means zero named evidence -- return None so the caller can try
    # ordinal/last instead, exactly like the single-segment path already
    # does on a plain no-match.
    resolved: dict[int, DisplayedProduct] = {}
    claimed: set[int] = set()
    any_match = False
    has_ambiguous = False
    has_unmatched = False

    for segment in segments:
        # Correction: a segment that IS a positional/last marker word
        # ("first", "last", "الاخير", a bare digit) is never a product-name
        # candidate here -- skip it entirely (neither a match nor a
        # failure). A PURELY positional multi-item message ("first and
        # last") is then left with any_match=False and correctly returns
        # None below, deferring to _resolve_product_selection()'s own
        # ordinal/last handling instead of being fuzzy-matched against an
        # unrelated product by coincidence. A MIXED message (some segments
        # positional, some named) is handled entirely separately, one level
        # up, by _resolve_product_selection() itself -- this function is
        # never called at all for that case (see its own docstring).
        if _is_pure_positional_word(segment):
            continue

        # Each segment names a DISTINCT requested item -- a product another
        # segment already claimed is removed from THIS segment's own pool,
        # so it can never be "reclaimed" by a weaker, coincidental fuzzy
        # score on a segment that actually names something else entirely
        # (see _match_segment_to_product()'s own docstring for the WRatio
        # quirk this guards against).
        pool = [p for p in recent_shown if p.id not in claimed]
        product, reason = _match_segment_to_product(segment, pool)
        if reason == "ok":
            resolved[product.id] = product
            claimed.add(product.id)
            any_match = True
        elif reason == "ambiguous":
            has_ambiguous = True
        else:
            has_unmatched = True

    if not any_match:
        return None
    if has_ambiguous:
        return [], "ambiguous"
    if has_unmatched:
        return [], "not_found"
    return list(resolved.values()), "ok"


def _resolve_product_selection(
    current_shown: list[DisplayedProduct], recent_shown: list[DisplayedProduct],
    action: Action, raw_message: str,
) -> tuple[list[DisplayedProduct], str]:
    """Sprint 6 product-selection task, generalized for cross-turn multi-
    item reference resolution: the ONE resolver every caller uses --
    plain shown-product selection, direct checkout (capped to one product
    by the caller), and cart add/remove/update-quantity/checkout-subset
    (against the cart pool instead of shown products) -- purely
    deterministic, never the wider catalog and never an LLM-chosen id.
    Returns (products, reason); reason is "ok" on success (products is
    always non-empty then -- one entry for an ordinary selection, 2+ only
    when the raw message named that many complete shown/cart titles at
    once, or positions at once via selection_ordinals), else one of
    "no_context"/"out_of_range"/"ambiguous"/"not_found" with an empty list,
    so the caller can pick the exact honest wording (never a guess, never a
    partial mutation/checkout of only SOME of what was named).

    Two DISTINCT authorities, never mixed: `current_shown` (the current
    page -- the MOST RECENT set_last_shown_products call) is the only
    source for POSITIONAL/vague references ("first one"/"third one"/"last
    one"/"it") -- a position must never reach into an older page the user
    isn't looking at anymore. `recent_shown` (a bounded window of the last
    few pages, deduplicated by id -- see SessionActivityStore.
    recent_shown_products()) is what NAMED/exact references (a raw-message
    complete title, or the LLM's selection_text) may search -- real bug:
    "recommend fish" (Sardine Fish shown) then "recommend chips" (Chipsy
    shown, superseding Sardine as the current page) then "checkout to
    Chipsy and Sardine Fish" used to forget Sardine entirely because
    selection only ever consulted the CURRENT page.

    The user's own RAW message is checked for shown products' exact/near
    names FIRST (see _match_named_products()), ahead of every LLM-derived
    signal -- the LLM can correctly notice a turn IS a selection while
    still mis-copying WHICH product into Action.selection_text (real
    symptom: the user typed the exact "...Spicy Tortilla..." title
    verbatim, but selection_text came back as the OTHER shown "...Whole
    Wheat Tortilla..." title, and the resolver selected -- and later
    checked out -- the wrong real product). A complete shown title clearly
    present in the raw text is stronger evidence than anything the LLM
    reports about itself. A single named reference resolves outright when
    exactly one product matches; an explicit MULTI-item raw message ("buy X
    and Y", now also across turns) requires EVERY named item to resolve to
    exactly one product each (one unresolved/ambiguous item fails the
    WHOLE request -- never a partial checkout of only some of what was
    named) -- a bare brand or generic word is never enough since a match
    must equal a whole shown name (or, for a proven multi-item message
    only, a confident fuzzy match), not a fragment of one. selection_text
    only ever resolves to ONE product, scoped to the CURRENT page only.
    selection_ordinals may resolve to ONE OR MORE products (cart/direct-
    checkout architecture task correction: "first and third" -> [1, 3]),
    still scoped to the CURRENT page only. selection_last/selection_ordinals
    are trusted ONLY when the RAW MESSAGE itself actually contains the
    corresponding positional evidence (see
    _raw_message_supports_last()/_raw_message_supports_ordinal()) -- never
    validated against another LLM field, since that would still leave
    product identity entirely LLM-controlled. Real confirmed symptom: "yes
    i want to buy it" against several shown Pepsi products returned
    selection_last=True (and an ambiguous selection_text "Pepsi") even
    though the user's own words carry no positional evidence at all. When
    a position isn't raw-supported, or is out of range, the WHOLE request
    fails immediately (never falls back to trusting selection_text instead,
    and never returns the OTHER positions that did resolve -- a partially
    fabricated/out-of-range multi-position request means the whole turn's
    specificity is suspect, so only an honest "ambiguous"/"out_of_range" is
    returned, not a guess or a partial result). The one exception: a single
    CURRENTLY shown product is trivially "the last one" by elimination --
    not LLM trust, just the absence of any other possible answer.

    MIXED named+positional requests (correction: "first and Sardine Fish",
    "Pepsi and third", "الاول والسردين") are resolved by the dedicated
    block immediately below, BEFORE _match_named_products() ever runs --
    real bug this fixes: the named path used to return [Sardine], "ok"
    outright the instant ANY segment matched a name, so the positional
    branch further down was never even reached, silently dropping "first".
    Each segment still resolves against its OWN correct authority (a
    positional segment against current_shown, a named segment against
    recent_shown -- the "two authorities, never mixed" rule above still
    holds per-segment), combined into ONE all-or-nothing result. A message
    that is ENTIRELY positional words ("first and last") or entirely named
    ("Pepsi and Sardine") is NOT mixed and falls through unchanged to the
    existing single-purpose logic below."""
    if not current_shown:
        return [], "no_context"

    segments = _split_named_items(raw_message, recent_shown)
    if len(segments) >= 2:
        positional_flags = [_is_pure_positional_word(s) for s in segments]
        if any(positional_flags) and not all(positional_flags):
            resolved: dict[int, DisplayedProduct] = {}
            # named_claimed tracks ONLY products a NAMED segment already
            # matched (the WRatio-quirk protection _match_segment_to_
            # product() callers rely on elsewhere) -- deliberately NOT
            # populated by a positional segment's own resolution. A
            # position and a name can legitimately refer to the SAME
            # product ("first, Pepsi and last" where position 1 IS Pepsi)
            # -- excluding a positionally-claimed product from the named
            # pool would wrongly make that later "Pepsi" segment fail to
            # find itself. Final dedup-by-id (`resolved[product.id] = ...`)
            # already collapses a genuine overlap into one line regardless.
            named_claimed: set[int] = set()
            for segment, is_positional in zip(segments, positional_flags):
                if is_positional:
                    idx, reason = _positional_index(segment, current_shown, raw_message)
                    if reason != "ok":
                        return [], reason
                    product = current_shown[idx]
                else:
                    pool = [p for p in recent_shown if p.id not in named_claimed]
                    product, reason = _match_segment_to_product(segment, pool)
                    if reason != "ok":
                        return [], reason
                    named_claimed.add(product.id)
                resolved[product.id] = product
            return list(resolved.values()), "ok"

    named = _match_named_products(raw_message, recent_shown)
    if named is not None:
        return named

    if action.selection_ordinals or action.selection_last:
        # Correction: selection_ordinals and selection_last MAY coexist
        # ("first and last", "first, third and last", "الاول والاخير") --
        # each is validated against its OWN raw evidence independently,
        # then COMBINED into one all-or-nothing set, never one silently
        # overriding/dropping the other (real bug: selection_last used to
        # return unconditionally before selection_ordinals was ever
        # checked, so "first and last" silently lost "first" entirely).
        # Dedupe by 0-based index, preserving first-seen order -- the same
        # position stated twice (or "last" coinciding with a stated final
        # ordinal) must never produce two lines. Each stated ordinal needs
        # its OWN raw-text evidence; the first missing or out-of-range
        # position fails the ENTIRE request (never a partial resolve of
        # only the positions that did check out).
        indices: list[int] = []
        for ordinal in dict.fromkeys(action.selection_ordinals or ()):
            if not _raw_message_supports_ordinal(raw_message, ordinal):
                return [], "ambiguous"
            indices.append(ordinal - 1)
        if action.selection_last:
            # A single CURRENTLY shown product is trivially "the last one"
            # by elimination -- no real ambiguity is possible regardless of
            # the raw message, so this is not "trusting the LLM," just
            # there being only one possible answer.
            if len(current_shown) == 1 or _raw_message_supports_last(raw_message):
                indices.append(len(current_shown) - 1)
            else:
                return [], "ambiguous"
        indices = list(dict.fromkeys(indices))
        resolved: list[DisplayedProduct] = []
        for idx in indices:
            if idx < 0 or idx >= len(current_shown):
                return [], "out_of_range"
            resolved.append(current_shown[idx])
        return resolved, "ok"

    if action.selection_text:
        # Ordinary single fuzzy reference -- deliberately scoped to
        # CURRENT_SHOWN only (never the wider recent_shown pool), so an
        # unrelated similarly-named product from an older page can never
        # interfere with/false-ambiguate a plain current-page selection.
        # Cross-turn named reach is already covered above by
        # _match_named_products(), which works off the raw message
        # directly rather than this LLM-extracted field.
        text_norm = _normalize_field(action.selection_text)

        # An exact match against a shown product's own name always wins
        # outright, before fuzzy scoring runs -- fuzzy WRatio alone can
        # score several similarly-worded variants above threshold even
        # when one is a perfect match.
        exact_matches = [
            p for p in current_shown
            if text_norm in (_normalize_field(p.name_en), _normalize_field(p.name_ar))
        ]
        if len(exact_matches) == 1:
            return exact_matches, "ok"
        if len(exact_matches) > 1:
            return [], "ambiguous"

        matches = [p for p in current_shown if _selection_text_score(text_norm, p) >= _SELECTION_TEXT_FUZZY_THRESHOLD]
        if len(matches) == 1:
            return matches, "ok"
        if len(matches) > 1:
            return [], "ambiguous"
        return [], "not_found"

    return [], "not_found"  # defensive -- C3's own guard already ensures one field is set


def _trace_resolve(
    trace: RequestTrace, authority: str,
    current_shown: list[DisplayedProduct], recent_shown: list[DisplayedProduct],
    action: Action, raw_message: str,
) -> tuple[list[DisplayedProduct], str]:
    """Instructor-demo observability task: thin, read-only wrapper around
    _resolve_product_selection() -- records which product-resolution
    authority/pool a caller consulted and what it returned, purely for the
    human-readable trace (src/scenario/trace.py). Never changes the
    resolver's own logic/result; a drop-in replacement at each of its
    existing call sites."""
    products, reason = _resolve_product_selection(current_shown, recent_shown, action, raw_message)
    trace.resolution_authority = authority
    trace.resolution_product_ids = [p.id for p in products]
    trace.resolution_reason = reason
    return products, reason


def _trace_set_action(trace: RequestTrace, action: Action) -> None:
    """Instructor-demo observability task: records the already-structured
    Action fields (never LLM free-text/reasoning) that decided this turn --
    either a real understand() result or (A2.5c) a purely raw-text-derived
    synthetic Action."""
    trace.action_intent = action.intent.value
    trace.action_cart_action = action.cart_action
    trace.action_checkout_request = action.checkout_request
    trace.action_selection_ordinals = list(action.selection_ordinals) if action.selection_ordinals else None
    trace.action_selection_text = action.selection_text


def _selection_failure_message(message: str, reason: str, shown_count: int, *, cart: bool = False) -> str:
    """The exact honest wording for each non-"ok" _resolve_product_selection
    reason -- factored out so a standalone selection turn, a combined
    selection+checkout turn, and (cart/direct-checkout architecture task)
    cart add/remove/update/checkout-subset turns all share it. `cart=True`
    selects the cart-flavored wording (pool_kind = cart lines, not shown
    products) -- same reasons, same resolver, different pool/wording only."""
    if cart:
        if reason == "no_context":
            return messages.cart_no_items_response(message)
        if reason == "out_of_range":
            return messages.cart_selection_out_of_range_response(message, shown_count)
        if reason == "ambiguous":
            return messages.cart_selection_ambiguous_response(message)
        return messages.cart_selection_not_found_response(message)  # "not_found"
    if reason == "no_context":
        return messages.selection_no_display_context_response(message)
    if reason == "out_of_range":
        return messages.selection_out_of_range_response(message, shown_count)
    if reason == "ambiguous":
        return messages.selection_ambiguous_response(message)
    return messages.selection_not_found_response(message)  # "not_found"
