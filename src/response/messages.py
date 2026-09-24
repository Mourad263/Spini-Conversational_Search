"""Sprint 4 deterministic response wording. No LLM, no free-form text
generation, no template framework -- every function returns a fixed
string, optionally parameterized only by already-resolved, real values
(never an invented product fact). src/scenario/orchestrator.py composes
these into a turn's supplementary `message` field.

Sprint 6 Task 3A addition: `build_assistant_message()` is a SEPARATE,
always-populated conversational reply (the new `assistant_message` API
field) -- it does not replace or read `message`, which keeps its existing
meaning (a supplementary note, often null) unchanged. Bilingual (EN/AR),
selected by a dominant-script check on the CURRENT user message
(is_arabic_dominant()) -- no language-detection library, no LLM call. Every string
is a fixed template; nothing here ever inserts a product/brand/category
name, a price, or a count, so there is nothing here to hallucinate.
"""

from __future__ import annotations

from src.ingestion.text_normalize import is_arabic_dominant


def off_topic_refocus(customer_service_tone: bool = False) -> str:
    if customer_service_tone:
        return "No problem -- I can help you search for products. What are you looking for?"
    return "I can help you search for products in our catalog -- what are you looking for?"


def recommendation_needs_clarification() -> str:
    return "Tell me what product, category, or brand you're looking for and I can help."


def recommendation_relaxation_exhausted() -> str:
    return (
        "I've already widened the price range a few times. Try giving me a new budget "
        "or a different filter."
    )


def rapid_repeat_blocked() -> str:
    return "You've sent that same request a few times in a row -- let me know if you'd like something different."


def zero_result(likely_out_of_catalog: bool) -> str:
    # Deliberately never a categorical claim ("we don't carry that") -- the
    # deterministic likely_out_of_catalog signal is corroborating evidence,
    # not proof (Sprint 4 design pass).
    if likely_out_of_catalog:
        return "I couldn't find that in our current catalog."
    return "I couldn't find any matching products for that search."


def customer_service_acknowledgment() -> str:
    return "Sorry for the trouble -- here's what I found:"


def turn_cap_suggestion() -> str:
    return "This conversation has gotten long -- you can reset the search if you'd like to start fresh."


def recommendation_band_note() -> str:
    return "I widened the price range slightly to show you more matching options."


def understanding_failed() -> str:
    return "Something went wrong understanding that -- please try again."


def unresolved_exclusion_note(excluded: str) -> str:
    # Sprint 6 Finding B: an explicit free-from request with no matching
    # catalog category must never be silently ignored -- this is the
    # user-visible admission that it wasn't applied.
    return f"Note: I couldn't filter out \"{excluded}\" specifically, so these results may include it."


# ---------------------------------------------------------------------
# Task 3A: deterministic conversational response layer.
# ---------------------------------------------------------------------

GREETING_EN = "Hello! I'm Spini. How can I help you?"
GREETING_AR = "مرحباً! أنا Spini، كيف يمكنني مساعدتك؟"


def greeting(arabic: bool = False) -> str:
    """Reusable initial greeting. Sprint 6 social fast-path task: now
    actually wired in, via social_greeting_response() below."""
    return GREETING_AR if arabic else GREETING_EN


# ---------------------------------------------------------------------
# Sprint 6: social fast-path responses (greeting/how-are-you/gratitude).
# No LLM call for these -- src/scenario/social.py classifies the message
# deterministically before understand() ever runs, and these functions
# supply the fixed reply. Same is_arabic_dominant()-on-current-message convention
# as build_assistant_message()/memory_recall_answer() above.
# ---------------------------------------------------------------------

HOW_ARE_YOU_EN = "I'm doing well, thanks for asking! How can I help you?"
HOW_ARE_YOU_AR = "بخير، كيف يمكنني مساعدتك؟"

_GRATITUDE_EARLY = (
    "You're welcome! How can I help you?",
    "العفو! كيف يمكنني مساعدتك؟",
)
_GRATITUDE_LATE = (
    "You're welcome! I hope I was able to help.",
    "العفو! أتمنى أكون قدرت أساعدك.",
)


def social_greeting_response(user_message: str) -> str:
    return GREETING_AR if is_arabic_dominant(user_message) else GREETING_EN


def social_how_are_you_response(user_message: str) -> str:
    return HOW_ARE_YOU_AR if is_arabic_dominant(user_message) else HOW_ARE_YOU_EN


NAME_QUESTION_EN = "My name is Spini."
NAME_QUESTION_AR = "اسمي Spini."


def social_name_response(user_message: str) -> str:
    return NAME_QUESTION_AR if is_arabic_dominant(user_message) else NAME_QUESTION_EN


def social_gratitude_response(user_message: str, had_prior_activity: bool) -> str:
    """`had_prior_activity` (Sprint 6 social fast-path task): a simple,
    existing-signal read of whether this session already had at least one
    real shopping/memory turn (src/scenario/session_activity.py's
    `turns`) -- not sentiment analysis, not an LLM call. Distinguishes
    "thanks" near the start of a conversation from "thanks" after Spini
    actually helped with something."""
    en, ar = _GRATITUDE_LATE if had_prior_activity else _GRATITUDE_EARLY
    return ar if is_arabic_dominant(user_message) else en


# ---------------------------------------------------------------------
# Sprint 6: proactive recommendation offer + pending-action response. No
# LLM call -- src/scenario/orchestrator.py appends the offer to the
# existing search reply, and src/scenario/offer_response.py classifies a
# following accept/reject deterministically. Same is_arabic_dominant() convention.
# ---------------------------------------------------------------------

RECOMMENDATION_OFFER_EN = "Would you like me to recommend some alternatives?"
RECOMMENDATION_OFFER_AR = "هل تريد أن أرشح لك بعض البدائل؟"

_RECOMMENDATION_OFFER_DECLINED = (
    "No problem. What else can I help you find?",
    "تمام، تحب أدور لك على حاجة تانية؟",
)


def recommendation_offer_prompt(user_message: str) -> str:
    return RECOMMENDATION_OFFER_AR if is_arabic_dominant(user_message) else RECOMMENDATION_OFFER_EN


def recommendation_offer_declined_response(user_message: str) -> str:
    en, ar = _RECOMMENDATION_OFFER_DECLINED
    return ar if is_arabic_dominant(user_message) else en


# ---------------------------------------------------------------------
# Sprint 6: product selection from previously-shown products only. No LLM
# call for the wording -- src/scenario/orchestrator.py resolves the real
# product (or the reason it couldn't) deterministically against
# src/scenario/session_activity.py's last_shown_products, and these
# functions supply the fixed reply. Same is_arabic_dominant() convention.
# ---------------------------------------------------------------------

def product_selected_response(user_message: str, products: list[tuple[str, str]]) -> str:
    """`products`: (name_en, name_ar) per selected product -- 2+ only when
    the user's own message named that many complete shown-product titles
    at once."""
    arabic = is_arabic_dominant(user_message)
    names = [ar if arabic else en for en, ar in products]
    joined = "، ".join(names) if arabic else ", ".join(names)
    return f"تمام، اخترت: {joined}." if arabic else f"You selected: {joined}."


def selection_no_display_context_response(user_message: str) -> str:
    if is_arabic_dominant(user_message):
        return "لسه معرضتش أي منتجات. دور على منتج الأول."
    return "I haven't shown any products yet. Search for a product first."


def selection_out_of_range_response(user_message: str, shown_count: int) -> str:
    if is_arabic_dominant(user_message):
        return f"أنا عرضت {shown_count} منتجات بس. اختار واحد منهم."
    return f"I only showed {shown_count} products. Please choose one of those."


def selection_ambiguous_response(user_message: str) -> str:
    if is_arabic_dominant(user_message):
        return "لقيت أكتر من منتج مطابق من اللي عرضته. تقدر تحدد بالرقم أو بالاسم الكامل؟"
    return "I found more than one match among what I showed you. Could you specify by number or full name?"


def selection_not_found_response(user_message: str) -> str:
    if is_arabic_dominant(user_message):
        return "ده مش من المنتجات اللي عرضتها."
    return "That wasn't one of the products I showed you."


# ---------------------------------------------------------------------
# Sprint 6: checkout initiation. No LLM call for the wording, no order/
# payment claim -- src/scenario/orchestrator.py starts a checkout DRAFT
# only from real, already-selected product(s) (src/scenario/
# session_activity.py's selected_products). Same is_arabic_dominant()
# convention as the rest of this module.
# ---------------------------------------------------------------------

def checkout_price_unavailable_response(user_message: str, products: list[tuple[str, str]]) -> str:
    """`products`: (name_en, name_ar) of the shown product(s) whose
    effective price is unknown (src/checkout/demo_order.py's
    effective_price() returned None) -- checkout is refused before any
    CheckoutForm is ever built, never a partial checkout of only the
    priced products."""
    arabic = is_arabic_dominant(user_message)
    names = [ar if arabic else en for en, ar in products]
    joined = "، ".join(names) if arabic else ", ".join(names)
    if arabic:
        return f"معلش، سعر {joined} مش متوفر دلوقتي، فمش هقدر أكمل الطلب بيه."
    return f"Sorry, the price for {joined} isn't available right now, so I can't check out with it."


def checkout_started_response(user_message: str, lines: list[tuple[str, str, int]]) -> str:
    """Cart/direct-checkout architecture task (generalized from
    (name_en, name_ar)-only): `lines` is (name_en, name_ar, quantity) per
    checkout line -- direct (always exactly one line, quantity 1) or
    cart-sourced (all cart lines, or a resolved subset, quantity copied
    from the cart). Explicitly states the FINAL checkout selection AND
    quantities before the form -- the user must never have to guess which
    products/quantities the resolver picked (this is the ONE checkout
    confirmation both flows converge on)."""
    arabic = is_arabic_dominant(user_message)
    if arabic:
        rows = "\n".join(f"{i}. {name_ar} × {qty}" for i, (_, name_ar, qty) in enumerate(lines, start=1))
        return (
            f"تمام! بتشتري:\n{rows}\n\n"
            "راجع الكميات بالأسفل.\nطريقة الدفع: الدفع عند الاستلام."
        )
    rows = "\n".join(f"{i}. {name_en} × {qty}" for i, (name_en, _, qty) in enumerate(lines, start=1))
    return (
        f"You're checking out:\n{rows}\n\n"
        "Review the quantities below.\nPayment is Cash on Delivery."
    )


# ---------------------------------------------------------------------
# Cart/direct-checkout architecture task: deterministic cart wording. No
# LLM call, no invented product/price/quantity -- src/scenario/
# orchestrator.py resolves cart membership/quantities deterministically
# against SessionActivity.cart, and these functions only render already-
# known real values. Same is_arabic_dominant() convention as the rest of
# this module.
# ---------------------------------------------------------------------

def cart_empty_response(user_message: str) -> str:
    if is_arabic_dominant(user_message):
        return "السلة فاضية."
    return "Your cart is empty."


def cart_view_response(user_message: str, lines: list[tuple[str, str, int, float]]) -> str:
    """`lines`: (name_en, name_ar, quantity, unit_price) per cart line.
    unit_price is always a real, known src/pricing.py effective_price() --
    cart/direct-checkout architecture task (correction): an unpriced
    product can no longer enter the cart at all (see orchestrator.py's
    _unpriced()), so every line always has a real price/line-total and the
    total is always the REAL full cart total, never a known-price
    subset."""
    if not lines:
        return cart_empty_response(user_message)
    arabic = is_arabic_dominant(user_message)
    total = 0.0
    rows = []
    for i, (name_en, name_ar, qty, unit_price) in enumerate(lines, start=1):
        name = name_ar if arabic else name_en
        line_total = unit_price * qty
        total += line_total
        rows.append(f"{i}. {name} × {qty} -- {unit_price:g} × {qty} = {line_total:g}")
    body = "\n".join(rows)
    if arabic:
        return f"السلة:\n{body}\n\nالإجمالي: {total:g}"
    return f"Your cart:\n{body}\n\nCart total: {total:g}"


def cart_add_price_unavailable_response(user_message: str, products: list[tuple[str, str]]) -> str:
    """Cart/direct-checkout architecture task (correction): an unpriced
    product (src/pricing.py's effective_price() returned None) must never
    enter the cart -- refused here, before add_to_cart() is ever called,
    same "block before mutation, never partial" rule checkout's own
    unknown-price guard uses. `products`: (name_en, name_ar) of the
    unpriced item(s) -- for a multi-item add, ANY unpriced product blocks
    the WHOLE add (all-or-nothing), never a partial cart mutation."""
    arabic = is_arabic_dominant(user_message)
    names = [ar if arabic else en for en, ar in products]
    joined = "، ".join(names) if arabic else ", ".join(names)
    if arabic:
        return f"معلش، سعر {joined} مش متوفر دلوقتي، فمش أقدر أضيفه للسلة."
    return f"Sorry, the price for {joined} isn't available right now, so I can't add it to your cart."


def cart_add_confirmation_response(user_message: str, added: list[tuple[str, str, int]]) -> str:
    """`added`: (name_en, name_ar, resulting_quantity) per product just
    added/incremented."""
    arabic = is_arabic_dominant(user_message)
    if arabic:
        rows = "، ".join(f"{ar} × {qty}" for _, ar, qty in added)
        return f"تمام، ضفت للسلة: {rows}."
    rows = ", ".join(f"{en} × {qty}" for en, _, qty in added)
    return f"Added to your cart: {rows}."


def cart_remove_confirmation_response(user_message: str, removed: list[tuple[str, str]]) -> str:
    arabic = is_arabic_dominant(user_message)
    if arabic:
        rows = "، ".join(ar for _, ar in removed)
        return f"تمام، شلت من السلة: {rows}."
    rows = ", ".join(en for en, _ in removed)
    return f"Removed from your cart: {rows}."


def cart_clear_confirmation_response(user_message: str) -> str:
    if is_arabic_dominant(user_message):
        return "تمام، فضيت السلة."
    return "Your cart has been cleared."


def cart_quantity_updated_response(user_message: str, name_en: str, name_ar: str, quantity: int) -> str:
    if is_arabic_dominant(user_message):
        return f"تمام، خليت {name_ar} × {quantity}."
    return f"Updated {name_en} to × {quantity}."


def cart_quantity_missing_response(user_message: str) -> str:
    if is_arabic_dominant(user_message):
        return "عايز الكمية تبقى كام؟"
    return "What quantity would you like?"


def cart_no_items_response(user_message: str) -> str:
    return cart_empty_response(user_message)


def cart_selection_out_of_range_response(user_message: str, cart_count: int) -> str:
    if is_arabic_dominant(user_message):
        return f"في السلة {cart_count} منتجات بس. اختار واحد منهم."
    return f"You only have {cart_count} products in your cart. Please choose one of those."


def cart_selection_ambiguous_response(user_message: str) -> str:
    if is_arabic_dominant(user_message):
        return "لقيت أكتر من منتج مطابق في السلة. تقدر تحدد بالرقم أو بالاسم الكامل؟"
    return "I found more than one matching product in your cart. Could you specify by number or full name?"


def cart_selection_not_found_response(user_message: str) -> str:
    if is_arabic_dominant(user_message):
        return "ده مش موجود في السلة."
    return "That's not in your cart."


def cart_add_vague_reference_response(user_message: str) -> str:
    """A cart_action="add" turn whose only "evidence" is a vague singular
    reference ("it"/"this"/"ده"/"دي") with 2+ shown products currently on
    screen -- see src/scenario/orchestrator.py's
    _raw_message_has_vague_singular_reference()."""
    if is_arabic_dominant(user_message):
        return "تحب تضيف أنهي منتج للسلة؟ قولّي الاسم أو الرقم."
    return "Which product would you like to add to your cart? You can tell me the product name or number."


def multi_direct_needs_cart_response(user_message: str, products: list[tuple[str, str]]) -> str:
    """The MULTI_DIRECT_REQUIRING_CART prompt: 2+ products named directly
    for checkout (without the cart) are never checked out in one line --
    the cart is the only durable authority for buying multiple products
    together. Lists what was safely resolved and asks whether to add it to
    the cart; the actual add only happens on an explicit accept (see
    src/scenario/orchestrator.py's A2.5 "multi_add_to_cart_pending" branch,
    reusing the SAME accept/reject classifier as a recommendation offer)."""
    arabic = is_arabic_dominant(user_message)
    if arabic:
        rows = "\n".join(f"{i}. {ar}" for i, (_, ar) in enumerate(products, start=1))
        return (
            "عشان تشتري أكتر من منتج مع بعض، لازم الأول تضيفهم للسلة.\n\n"
            f"لقيت:\n{rows}\n\nتحب أضيفهم للسلة؟"
        )
    rows = "\n".join(f"{i}. {en}" for i, (en, _) in enumerate(products, start=1))
    return (
        "To checkout multiple products together, add them to your cart first.\n\n"
        f"I found:\n{rows}\n\nWould you like me to add them to your cart?"
    )


def multi_add_to_cart_declined_response(user_message: str) -> str:
    if is_arabic_dominant(user_message):
        return "تمام، مفيش حاجة اتضافت للسلة."
    return "No problem, nothing was added to your cart."


def cart_checkout_cancelled_response(user_message: str) -> str:
    """Cart/direct-checkout architecture task (correction): deterministic
    cancellation of a pending cart-checkout scope clarification ("no"/
    "cancel"/"مش دلوقتي" -- see src/scenario/offer_response.py's bounded
    REJECT set) -- no checkout draft is opened, the cart itself is
    untouched."""
    if is_arabic_dominant(user_message):
        return "تمام، سيبنا الحساب دلوقتي."
    return "No problem, checkout cancelled."


def cart_checkout_empty_response(user_message: str) -> str:
    if is_arabic_dominant(user_message):
        return "السلة فاضية. تحب تضيف إيه؟"
    return "Your cart is empty. What would you like to add?"


def cart_checkout_clarify_response(user_message: str, lines: list[tuple[str, str, int]]) -> str:
    """CASE 3 (cart has multiple products, bare "checkout"): lists the cart
    and asks all-vs-subset, never assumes "all". `lines`: (name_en,
    name_ar, quantity) per cart line."""
    arabic = is_arabic_dominant(user_message)
    if arabic:
        rows = "\n".join(f"{i}. {ar} × {qty}" for i, (_, ar, qty) in enumerate(lines, start=1))
        return f"في السلة {len(lines)} منتجات:\n{rows}\n\nتحب تحاسب على كله ولا تختار منتجات معينة؟"
    rows = "\n".join(f"{i}. {en} × {qty}" for i, (en, _, qty) in enumerate(lines, start=1))
    return (
        f"You have {len(lines)} products in your cart:\n{rows}\n\n"
        "Would you like to checkout all of them or choose specific products?"
    )


def checkout_order_submitted_response(arabic: bool, lines: list[tuple[str, str, int]]) -> str:
    """Sprint 6 checkout-submission task: the deterministic confirmation
    for a successfully persisted DemoOrder. `arabic` is decided by the
    caller (src/scenario/checkout_submission.py) from the session's own
    conversational history via is_arabic_dominant() -- never from
    customer-submitted PII (name/address), which carries no signal about
    which language the CHAT itself is in. `lines`: (name_en, name_ar,
    quantity) per order line. Deliberately never claims payment/delivery/
    real fulfillment -- see this module's docstring."""
    if arabic:
        items = "\n".join(f"{name_ar} × {qty}" for _, name_ar, qty in lines)
        return (
            f"تم تسجيل طلبك بنجاح.\n\n"
            f"{items}\nطريقة الدفع: الدفع عند الاستلام\n\n"
            "تم استلام بيانات الطلب وهو جاهز لمرحلة التوصيل."
        )
    items = "\n".join(f"{name_en} × {qty}" for name_en, _, qty in lines)
    return (
        f"Your order has been submitted successfully.\n\n"
        f"{items}\nPayment: Cash on Delivery\n\n"
        "Your order is ready for delivery processing."
    )


# (english, arabic) -- one or two clear phrases per state, no alternates,
# no randomness: deterministic output, easy to test.
_ASSISTANT_MESSAGES: dict[str, tuple[str, str]] = {
    "search_success": (
        "Sure, I found these products for you.",
        "أكيد، لقيت لك المنتجات دي.",
    ),
    "search_and_recommendations": (
        "Sure, I found these products for you, and I also found some options you may like.",
        "أكيد، لقيت لك المنتجات دي وكمان شوية اختيارات ممكن تعجبك.",
    ),
    "recommendation_success": (
        "Here are some other options you may like.",
        "دي شوية اختيارات تانية ممكن تعجبك.",
    ),
    "recommendation_none": (
        "You've already seen the available matching options.",
        "إنت شفت بالفعل الاختيارات المتاحة المطابقة لطلبك.",
    ),
    "zero_result": (
        "I couldn't find products matching those filters.",
        "ملقتش منتجات مطابقة للفلاتر دي.",
    ),
    "out_of_catalog": (
        "I couldn't find a matching product in the current catalog.",
        "ملقتش منتج مطابق في الكتالوج الحالي.",
    ),
    "off_topic": (
        "I can help you search for and explore products in the store.",
        "أقدر أساعدك تدور على المنتجات وتستكشف اختيارات من المتجر.",
    ),
    "reset": (
        "Sure, I've cleared your current search. What would you like to find?",
        "تمام، بدأت لك بحث جديد. تحب تدور على إيه؟",
    ),
    "add_filter": (
        "Sure, I've updated your search with that filter.",
        "تمام، ضفت الفلتر للبحث.",
    ),
    "modify_filter": (
        "Sure, I've updated your search.",
        "تمام، عدّلت البحث.",
    ),
    "remove_filter": (
        "Sure, I've removed that filter.",
        "تمام، شلت الفلتر.",
    ),
    "blocked": (
        "I can't process that request. Please try a normal product search.",
        "مش هقدر أنفذ الطلب ده. جرّب تسألني عن منتج بشكل عادي.",
    ),
    "error": (
        "Something went wrong understanding that -- please try again.",
        "حصل مشكلة في فهم طلبك -- جرّب تاني.",
    ),
    "recommendation_needs_clarification": (
        "Tell me what product, category, or brand you're looking for and I can help.",
        "قولي عايز منتج إيه أو أي فئة أو ماركة، وأنا هساعدك.",
    ),
}


# ---------------------------------------------------------------------
# Sprint 6: session-scoped conversational memory. No LLM call, no
# catalog/product lookup -- answers come only from data the session
# already holds (src/scenario/session_activity.py's unbounded `turns`
# log, and StateManager's own canonical, already-merged FilterSet for the
# "product" case). Read-only: calling this never mutates anything.
# ---------------------------------------------------------------------

_MEMORY_NOTHING_YET: dict[str, tuple[str, str]] = {
    "previous": (
        "I don't have anything from earlier in this conversation yet.",
        "لسه معنديش حاجة من قبل في المحادثة دي.",
    ),
    "first": (
        "I don't have anything from earlier in this conversation yet.",
        "لسه معنديش حاجة من قبل في المحادثة دي.",
    ),
    "product": (
        "You haven't mentioned a product, brand, or price yet.",
        "لسه معنديش منتج أو ماركة أو سعر قولت عليه.",
    ),
}


def _memory_price_phrase(price_min: float | None, price_max: float | None, arabic: bool) -> str | None:
    if price_min is not None and price_max is not None:
        return f"سعر بين {price_min:g} و {price_max:g}" if arabic else f"price between {price_min:g} and {price_max:g}"
    if price_max is not None:
        return f"سعر تحت {price_max:g}" if arabic else f"price under {price_max:g}"
    if price_min is not None:
        return f"سعر فوق {price_min:g}" if arabic else f"price over {price_min:g}"
    return None


def memory_recall_answer(
    user_message: str,
    kind: str,
    *,
    previous_message: str | None,
    first_message: str | None,
    category: str | None,
    brand: str | None,
    price_min: float | None,
    price_max: float | None,
) -> str:
    """Deterministic answer to a memory_recall Action (schema.py). `kind`
    is one of "previous"/"first"/"product" -- exactly the three values
    understand() may set. Language matches the CURRENT recall question,
    via the same is_arabic_dominant() check every other assistant_message uses;
    an honest "nothing yet" fallback when the requested information
    genuinely isn't there rather than a made-up answer."""
    arabic = is_arabic_dominant(user_message)

    if kind == "previous":
        if previous_message is None:
            return _MEMORY_NOTHING_YET["previous"][1 if arabic else 0]
        return f"قبل كده قولت: \"{previous_message}\"." if arabic else f'Earlier you said: "{previous_message}".'

    if kind == "first":
        if first_message is None:
            return _MEMORY_NOTHING_YET["first"][1 if arabic else 0]
        return (
            f"أول حاجة طلبتها في المحادثة دي كانت: \"{first_message}\"." if arabic
            else f'The first thing you asked for in this conversation was: "{first_message}".'
        )

    # kind == "product"
    parts = []
    if category is not None:
        parts.append(f"الفئة {category}" if arabic else f"category {category}")
    if brand is not None:
        parts.append(f"الماركة {brand}" if arabic else f"brand {brand}")
    price_phrase = _memory_price_phrase(price_min, price_max, arabic)
    if price_phrase is not None:
        parts.append(price_phrase)
    if not parts:
        return _MEMORY_NOTHING_YET["product"][1 if arabic else 0]
    joined = "، ".join(parts) if arabic else ", ".join(parts)
    return f"لحد دلوقتي قولت على {joined}." if arabic else f"So far you've mentioned {joined}."


def build_assistant_message(
    *,
    user_message: str,
    intent: str,
    recommendation_request: bool,
    recommendations_count: int,
    zero_result: bool,
    likely_out_of_catalog: bool,
    blocked: bool,
    needs_clarification: bool = False,
) -> str:
    """One deterministic, always-non-null conversational reply. Built only
    from already-known runtime signals -- no new LLM call, no product/
    brand/category/price/count ever inserted (see module docstring).

    Precedence (checked in this exact order -- states are mutually
    exclusive at each step, so only ONE branch below ever fires):
      1. blocked                          -- security wording wins outright
      2. intent == error                  -- understand() itself failed
      3. intent == off_topic              -- never discusses products/results
      4. intent == reset
      4. zero_result (out_of_catalog is the more specific sub-case)
      5. needs_clarification              -- recommendation asked with NO
         active search context at all (nothing to explore from) -- distinct
         from, and must not be confused with, "asked but nothing new" below
      6. recommendation_request           -- honesty rule: recommendations_
         count==0 NEVER produces "I found recommendations" wording, whether
         or not a fresh search also happened this turn
      7. plain per-intent search/filter success
    """
    arabic = is_arabic_dominant(user_message)

    if blocked:
        key = "blocked"
    elif intent == "error":
        key = "error"
    elif intent == "off_topic":
        key = "off_topic"
    elif intent == "reset":
        key = "reset"
    elif zero_result:
        key = "out_of_catalog" if likely_out_of_catalog else "zero_result"
    elif needs_clarification:
        key = "recommendation_needs_clarification"
    elif recommendation_request:
        if recommendations_count <= 0:
            key = "recommendation_none"
        elif intent == "no_op":
            key = "recommendation_success"
        else:
            key = "search_and_recommendations"
    elif intent in ("add_filter", "modify_filter", "remove_filter"):
        key = intent
    else:
        key = "search_success"

    en, ar = _ASSISTANT_MESSAGES[key]
    return ar if arabic else en
