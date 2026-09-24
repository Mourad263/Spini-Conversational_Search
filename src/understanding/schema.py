"""Understanding layer: LLM-facing structured output schema (Sprint 2,
tasks.md T024; responsibility boundary redrawn in the Sprint 6 semantic-role
refactor).

This is a SEMANTIC ROLE schema, not a resolved-filter schema. The LLM
says what each part of the message MEANS (this part names a product, this
part names a brand, this part is an exclusion); it never says which
canonical catalog row that corresponds to. Canonical grounding -- real
category slug, real brand slug, whether the thing exists at all -- stays
entirely deterministic, in src/validation/validate.py against the live
catalog.

Why the roles moved to the LLM: they were previously all crammed into one
unsegmented raw_query_text field, and a downstream fuzzy matcher had to
guess from character similarity alone whether an arbitrary word was a
product, a brand, or a category. That guess is not recoverable from
spelling -- "peas" and the real brand "Pears", or "paste" and the real
category "Pasta", are one character apart -- so the guess kept being
wrong and kept being patched with per-word blocklists. Role is a
LANGUAGE question (the LLM can answer it from context); identity is a
CATALOG question (only the database can answer it).
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, model_validator


class Intent(str, Enum):
    SEARCH = "search"                  # a new search, replacing any prior filters
    ADD_FILTER = "add_filter"          # add a requirement on top of active filters
    MODIFY_FILTER = "modify_filter"    # change the value of an already-active filter
    REMOVE_FILTER = "remove_filter"    # drop one specific active filter
    RESET = "reset"                    # clear everything, start over
    OFF_TOPIC = "off_topic"            # message has no product-search purpose at all
    NO_OP = "no_op"                    # a valid, understood conversational turn that carries
    # no canonical search-state transition (e.g. a bare "what do you recommend?" with
    # nothing else stated) -- distinct from OFF_TOPIC: this IS about the shopping
    # conversation, it just names no new value for State Manager to act on. Orchestration
    # (Sprint 4) reads the existing session state via StateManager.get_state() for these
    # turns rather than calling apply() -- see recommendation_request below.


class Facet(str, Enum):
    CATEGORY = "category"
    BRAND = "brand"
    PRICE = "price"


class Action(BaseModel):
    intent: Intent
    price_min: float | None = None
    price_max: float | None = None
    raw_query_text: str | None = None  # THE PRODUCT CONCEPT the user is asking for, in the
    # user's own spelling (typos preserved -- catalog-backed spelling correction happens
    # downstream in ranking.py, and "helpfully" fixing a typo here would bypass that real
    # evidence). Brand mentions no longer live here (see brand_text). Consumed by
    # validate() as the free-text/BM25 part of the search.
    category_hint: str | None = None  # a SEMANTIC category description in the user's own
    # words ("olive oil", "vegetables", "منظفات") -- explicitly NOT a canonical slug and
    # never treated as one: validate() tries to ground it against real canonical category
    # names and leaves category unresolved if it can't. Exists because the LLM can tell
    # "cucumber" (a vegetable the user wants) from "cucumber face wash" (a face wash
    # flavoured with cucumber) from context, which character-similarity matching cannot.
    brand_text: str | None = None  # the brand the user actually NAMED, verbatim, or null.
    # Only set when the message semantically refers to a brand ("from Heinz", "جهينة",
    # "got anything by Dove") -- never because a product word happens to resemble some
    # brand that might exist. validate() fuzzy-matches this against the real brand table;
    # because the ROLE is already known here, that match needs no semantic blocklist.
    free_from: str | None = None  # the thing the user wants EXCLUDED, as a bare concept
    # ("sugar", "لاكتوز", "gluten") when the message expresses a free-from/without
    # relation ("no added sugar", "بدون لاكتوز", "milk without lactose"). validate()
    # grounds it to a real "X Free" category if one genuinely exists, and otherwise
    # leaves it unresolved rather than collapsing the request into the POSITIVE concept.
    target_facet: Literal["category", "brand", "price"] | None = None  # which filter
    # remove_filter clears -- see llm.py. Values match Facet's own (Facet is a str Enum,
    # so `action.target_facet == Facet.CATEGORY` etc. still holds downstream). Inlined as
    # a Literal instead of the Facet enum type itself because the Groq structured-output
    # endpoint's stricter JSON-schema validator rejects a nullable field whose non-null
    # branch is a $ref (`anyOf: [{"$ref": ...Facet}, {"type": "null"}]` -- "anyOf branches
    # must be disambiguated via a required discriminator"); every OTHER optional field
    # here is a plain `anyOf: [{type: string/number}, {type: null}]` and was never
    # affected. A Literal's values are inlined directly in the schema, which satisfies
    # Groq's validator with no behavior change.
    # Only meaningful for remove_filter: that intent often carries no value to extract
    # at all ("شيل السعر" has no raw_query_text and no new price), so there's nothing
    # else in the Action for the State Manager (Sprint 3) to know WHICH active filter
    # to drop. Every other intent already carries an explicit value (a price, or a
    # raw_query_text that validate() resolves) that says which field changed, so this
    # field is left null for them.
    recommendation_request: bool = False  # Sprint 4: the message asks for an opinion/pick
    # ("what do you recommend?", "what's best?") rather than stating a requirement. This
    # is ORTHOGONAL to intent, not a replacement for it -- a recommendation ask can
    # coexist with a real, concrete search-state change in the SAME message ("what do
    # you recommend from Juhayna?" is still add/modify_filter with raw_query_text=
    # "Juhayna", PLUS this flag). When nothing concrete is stated at all, intent is
    # NO_OP and this flag is the only signal that a recommendation-flavored response
    # (asking for criteria) is owed instead of an ordinary empty turn. This system is
    # not a recommendation engine (FR-018) -- the flag exists to trigger a clarifying
    # response, never to make the LLM invent a specific product pick.
    customer_service_tone: bool = False  # Sprint 4: the message carries frustration/anger
    # ("I'm really annoyed, just show me milk under 50") -- ORTHOGONAL to intent, purely
    # a response-wording signal (a brief acknowledgment prefix). Must never change what
    # intent/price/raw_query_text/target_facet already resolve to on their own.
    memory_recall: Literal["previous", "first", "product"] | None = None  # Sprint 6
    # session-memory task: set only when the CURRENT message asks about the CONVERSATION
    # itself (what was said/searched before), never about the catalog. "previous" = the
    # message right before this one; "first" = the very first thing asked this session;
    # "product" = what product/brand/price was mentioned so far. Answered downstream,
    # deterministically, from data src/scenario/session_activity.py and
    # src/state_manager/state_manager.py already store -- no new search, no
    # RecommendationService, no second LLM call (see src/response/messages.py's
    # memory_recall_answer() and src/scenario/orchestrator.py's handle_message()).
    selection_ordinals: list[int] | None = None  # Sprint 6 product-selection task,
    # generalized (cart/direct-checkout architecture task correction) from a single
    # selection_ordinal to a LIST: one or more 1-based positions ("first"=[1],
    # "third"=[3], "first and third"=[1, 3], "number 5"=[5]) when the user refers to
    # PREVIOUSLY SHOWN/cart product(s) by position, never a new search. Which real
    # product(s) those actually are gets resolved deterministically downstream against
    # src/scenario/session_activity.py's last_shown_products or the cart -- this field
    # only signals the position(s); it never carries or invents a product id. Every
    # stated position must be an EXPLICIT number/ordinal word -- never a range, never
    # "the rest".
    selection_last: bool = False  # same idea as selection_ordinals, for "last one"/"آخر
    # واحد"/"الاخير" instead of an explicit number. MAY coexist with selection_ordinals
    # (correction) when the raw wording explicitly states both together -- "first and
    # last" -> selection_ordinals=[1], selection_last=true; "first, third and last" ->
    # selection_ordinals=[1, 3], selection_last=true -- the resolver combines them into
    # one all-or-nothing set rather than one silently overriding/dropping the other.
    # Mutually EXCLUSIVE with selection_text: set selection_text only when NEITHER a
    # position nor "last" was stated.
    selection_text: str | None = None  # a brief description/brand of a PREVIOUSLY SHOWN
    # product ("the Juhayna one", "the lactose-free one", "جهينة") when the user refers
    # to it by description rather than position. Differs from brand_text: this picks
    # among products ALREADY displayed this session, never starts a new brand search --
    # resolution only ever looks at the already-shown list's own name/brand text, never
    # the wider catalog.
    checkout_request: bool = False  # Sprint 6 checkout task; cart/direct-checkout
    # architecture task (correction, matches final llm.py SYSTEM_PROMPT semantics):
    # the message asks to FINALIZE a purchase/checkout/pay -- by position
    # ("checkout the third one"), by a vague reference ("I want to buy it",
    # "عايز اشتريه"), OR BY A SPECIFIC NAME ("buy Pepsi", "checkout Pepsi") -- or to
    # checkout the cart ("checkout", "checkout all"). An OPEN-ENDED new shopping
    # requirement stays a normal search instead ("I want to buy some milk" -- a broad
    # category, not a specific item reference -- this stays false). When set alongside
    # selection_ordinals/selection_last/selection_text, that field is only a CANDIDATE
    # reference in the user's own words, never a claim that it is really shown, in the
    # cart, or exists at all. Whether a real product is actually available to check
    # out -- one shown/cart product directly, or the session's cart (empty/one/many/a
    # named subset) -- is decided entirely deterministically downstream
    # (src/scenario/orchestrator.py); this field only signals the user's intent to
    # finalize, never a product choice, cart membership, or a completed order.
    cart_action: Literal["add", "view", "remove", "clear", "update_quantity"] | None = None
    # Cart/direct-checkout architecture task: ONE coherent commerce-action field
    # (never several independent booleans) for the user's cart intent, orthogonal to
    # `intent` exactly like checkout_request/memory_recall/selection_* above.
    # "add"/"remove"/"update_quantity" reuse selection_ordinals/selection_last/
    # selection_text for WHICH product(s) -- never a second product-reference
    # mechanism. "view"/"clear" need no product reference. Real product/cart
    # membership/price are always resolved deterministically downstream; this field
    # only signals which cart operation was asked for. Null for anything else,
    # including a plain "buy X"/"checkout X" (that is checkout_request +
    # selection_text -- a purchase candidate, never a cart action just because a
    # product was named) and a bare "checkout" (checkout_request, not this).
    quantity: int | None = None  # Cart/direct-checkout architecture task: an
    # explicit quantity NUMBER the user actually stated ("add 3 Pepsi to cart",
    # "make Pepsi quantity 3", "change the second item to 2") -- reused for both
    # cart_action="add" (how many to add) and cart_action="update_quantity" (the
    # new total). Never invented when no number was said; downstream validation
    # (>=1, whole number) is deterministic, matching src/checkout/demo_order.py's
    # existing validate_quantity() rule.

    @model_validator(mode="after")
    def _target_facet_only_meaningful_for_remove_filter(self) -> "Action":
        # target_facet says which filter a bare remove_filter turn refers to. The
        # model occasionally sets it on other intents too (e.g. add_filter for "add
        # something from Munchi") even though the prompt says it's remove_filter-only;
        # since it is genuinely meaningless there, enforce the invariant structurally
        # rather than trust every call to self-police it.
        if self.intent != Intent.REMOVE_FILTER and self.target_facet is not None:
            self.target_facet = None
        return self
