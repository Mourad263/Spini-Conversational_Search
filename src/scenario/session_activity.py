"""Sprint 4 runtime/conversational metadata -- session_id -> SessionActivity.

Deliberately separate from src/state_manager/state_manager.py's
StateManager, which remains the sole authoritative owner of canonical user
search state (FilterSet: category/brand/query_text/price_min/price_max).
Nothing here duplicates those fields. This store only tracks what the
future conversational route (Part 3) needs from src/scenario/guard.py
(rapid-repeat/turn-cap bookkeeping) and src/scenario/decision.py
(recommendation-relaxation bookkeeping): the bounded raw-text history
understand() uses for its continuity judgment (src/understanding/llm.py's
HISTORY_WINDOW), plus a cumulative turn counter and a relaxation-step
counter that are pure runtime metadata, never part of the user's stated
search criteria.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.understanding.llm import HISTORY_WINDOW  # noqa: E402

SESSION_TTL_SECONDS = 45 * 60  # lazy inactivity TTL (Sprint 6 lifecycle task) -- no
# background worker: checked only on the next real access (SessionActivityStore.get()/
# is_expired()), against monotonic seconds (the same clock src/scenario/guard.py's
# rapid-repeat window already uses, threaded through via handle_message's own `now`).
# Within the requested 30-60 minute range; a plain module constant, not configurable,
# since this project has no config surface for it yet.

_RECLAIM_CHECK_EVERY = 50  # abandoned-session reclaim task: per-id lazy expiry
# (get()/is_expired()) only ever revisits the SAME session_id that happens to be
# accessed -- an id that's simply never accessed again (e.g. "New Chat" moving to
# a brand new session_id) never gets reclaimed at all, growing _sessions forever.
# reclaim_abandoned() below fixes that, but a full scan of _sessions on EVERY
# request would be real unbounded-per-request work -- amortized instead: it only
# actually scans on the 1st real access and every _RECLAIM_CHECK_EVERY-th one
# after, so steady-state per-request cost is zero on the other N-1 turns while
# still bounding how many requests an abandoned session can outlive. No
# background thread/scheduler -- purely piggybacked on normal request traffic.


@dataclass
class TurnRecord:
    """One accepted conversational turn, kept for session-scoped memory
    (Sprint 6) -- deliberately NOT the LLM-context history, which stays
    hard-bounded to HISTORY_WINDOW. Only what's needed to answer "what did
    I say before"/"what was my first search": the raw text and the intent
    it resolved to. No product/brand/price copy here -- a "what did I
    mention" answer reads that from StateManager's own canonical,
    already-merged FilterSet instead of duplicating it (see
    src/response/messages.py's memory_recall_answer())."""

    message: str
    intent: str | None = None
    outcome: str | None = None  # Compact conversation-state/LLM-memory task: a
    # short, deterministic summary of what this turn actually did (see
    # src/scenario/llm_context.py's summarize_turn_outcome()) -- set AFTER
    # append_history() by handle_message()'s own trace wrapper, once the
    # turn's real route/response are known (append_history() itself runs too
    # early in the pipeline to know the eventual outcome). None for a turn
    # the caller didn't record an outcome for.


@dataclass
class PendingAction:
    """Sprint 6 recommendation-offer task: the smallest representation of
    "what Spini just asked" -- operational conversation state, not
    conversational memory (never read by memory_recall, never persisted
    beyond the session). `kind` is a plain string, not an enum, so a later
    feature (e.g. checkout confirmation) can reuse this exact shape by
    adding a new kind value rather than a parallel mechanism. Consumed
    EXACTLY once by the very next turn (SessionActivityStore.
    pop_pending_action()) regardless of whether that turn actually
    resolves it -- see orchestrator.py's A2.5 branch for why that is what
    keeps a stale offer from ever hijacking a later, unrelated turn.

    shown_product_ids: the real product ids from the search that produced
    this offer -- exactly what RecommendationService.recommend()'s
    `search_result_ids` needs, remembered here so accepting the offer
    never requires a second SearchAdapter search."""

    kind: str
    shown_product_ids: list[int] = field(default_factory=list)


@dataclass
class DisplayedProduct:
    """Sprint 6 product-selection task: the smallest real, already-fetched
    subset of a shown product needed to resolve "the Juhayna one"/"the
    third one" later -- deliberately NOT the full ProductOut shape (no
    url_key/sku duplicated here; the id is the authoritative key). Built
    directly from the SAME dict rows SearchAdapter/RecommendationService
    already returned this turn -- no new DB query to populate this.

    price/special_price (Sprint 6 checkout task): added so a later
    checkout-form draft always has REAL price data cached from the moment
    of selection (see SessionActivity.selected_products) without ever
    needing a second deterministic product lookup."""

    id: int
    name_en: str
    name_ar: str
    brand_normalized: str | None = None
    price: float | None = None
    special_price: float | None = None


@dataclass
class CartLine:
    """Cart/direct-checkout architecture task: a product snapshot (the same
    server-authoritative DisplayedProduct -- id/name/price come only from a
    real shown-product record, never a client or LLM value) plus how many.
    ONE small shape reused for two conceptually-adjacent but distinct
    pieces of state (SessionActivity.cart -- durable across the whole
    session -- and SessionActivity.selected_products -- the ephemeral
    checkout-in-progress line items), rather than tracking quantity twice
    with two parallel representations."""

    product: DisplayedProduct
    quantity: int = 1


@dataclass
class SessionActivity:
    history: list[str] = field(default_factory=list)  # raw user turns, oldest-first,
    # bounded to HISTORY_WINDOW -- see SessionActivityStore.append_history(). Deliberately
    # NOT the canonical FilterSet fields (category/brand/query_text/price_min/price_max);
    # those live only in StateManager.
    turns: list[TurnRecord] = field(default_factory=list)  # the SAME accepted turns as
    # `history`, appended at the same point (append_history), but deliberately UNBOUNDED --
    # session-scoped conversational memory (Sprint 6), distinct from the small window sent
    # to the LLM. Still session-scoped only: cleared by reset(), never persisted across
    # sessions, never sent to understand().
    last_message: str | None = None  # normalized text of the most recent request, for
    # src/scenario/guard.py's rapid-repeat check
    repeat_count: int = 0
    repeat_window_start: float = 0.0  # monotonic seconds
    turn_count: int = 0  # cumulative accepted conversational turns
    relaxation_step: int = 0  # src/scenario/decision.py's recommendation price-relaxation
    # step counter -- never the user's stated price
    last_active: float | None = None  # monotonic seconds of the last real access --
    # Sprint 6 lifecycle task's lazy-TTL clock (see SESSION_TTL_SECONDS); stamped only
    # by SessionActivityStore.get(session_id, now=...). None means "never stamped yet"
    # (a session touched only via bare get(session_id), e.g. some tests) -- deliberately
    # NOT 0.0: monotonic clocks are an arbitrary, often-large epoch (process/system
    # uptime), so treating an un-stamped session as "last active at time zero" would
    # make it look ancient -- and therefore already expired -- the instant a real `now`
    # is ever passed in, which is wrong for a session nothing has actually timed out.
    pending_action: PendingAction | None = None  # Sprint 6 recommendation-offer task --
    # see PendingAction. None whenever nothing is pending (the overwhelming majority of
    # turns); never long-term memory, never sent to understand().
    last_shown_products: list[DisplayedProduct] = field(default_factory=list)  # Sprint 6
    # product-selection task: the MOST RECENTLY displayed real, ordered product list --
    # either normal search results or recommendations, whichever was shown LAST (a
    # recommendation shown after search results supersedes them; see
    # SessionActivityStore.set_last_shown_products()). Authoritative for POSITIONAL/
    # vague references only ("first one"/"third one"/"last one"/"it") -- the LLM never
    # sees or invents this list, only signals which KIND of reference the user made
    # (schema.py's selection_* fields). Untouched by turns that show nothing new
    # (zero-result, out-of-catalog, memory recall, social, off-topic) -- the OLD list
    # simply survives those turns.
    recent_shown_pages: list[list[DisplayedProduct]] = field(default_factory=list)  #
    # cross-turn multi-product checkout task: a bounded window of the last few PAGES
    # set via set_last_shown_products (current page always == recent_shown_pages[-1]),
    # so an explicitly NAMED product ("Sardine Fish") stays checkout-selectable even
    # after a LATER turn shows different products and replaces last_shown_products --
    # positional references must never reach this far back (see
    # SessionActivityStore.recent_shown_products()/orchestrator.py's
    # _resolve_product_selection()), only last_shown_products above. Bounded to
    # HISTORY_WINDOW pages -- the SAME window understand() already treats as "recent"
    # conversational context, reused rather than inventing a new number.
    selected_products: list[CartLine] = field(default_factory=list)  # Cart/direct-
    # checkout architecture task (was list[DisplayedProduct] pre-cart): the CHECKOUT
    # SELECTION -- the exact product(s)+quantity being purchased NOW. Set ONLY at the
    # moment a checkout draft actually opens (never merely from an acknowledged
    # selection -- see orchestrator.py's C3 vs C4), by exactly one of two paths: (a)
    # DIRECT checkout -- always exactly ONE entry, quantity 1, resolved from shown-
    # product authority (last_shown_products/recent_shown_pages) -- 2+ named products
    # can NEVER reach this path (see orchestrator.py's MULTI_DIRECT_REQUIRING_CART);
    # (b) CART checkout -- one entry per checked-out cart line (all cart lines, or a
    # resolved subset), quantity copied from the cart line. Never invented, never
    # catalog-searched, never a partial/mixed-authority list -- set only by
    # SessionActivityStore.set_selected_products().
    checkout_cart_ids: list[int] = field(default_factory=list)  # Cart/direct-checkout
    # architecture task: empty for a DIRECT checkout (cart untouched on success);
    # non-empty for a CART checkout (all or subset) -- exactly the product ids among
    # `cart` that this open draft came from, so checkout_submission.py knows which
    # (if any) cart lines to remove, and ONLY after successful persistence (never at
    # initiation -- see module docstring's ordering guarantee).
    checkout_active: bool = False  # Sprint 6 checkout task: true only while a checkout
    # draft is open for selected_products -- cleared on RESET, session expiry, and a
    # new successful normal search/topic switch (see orchestrator.py), never carried
    # across those as stale state.
    cart: list[CartLine] = field(default_factory=list)  # Cart/direct-checkout
    # architecture task: the DURABLE cart -- products the user explicitly asked to
    # save for later purchase (concept B, distinct from `selected_products`/concept C
    # and from last_shown_products/concept A). Survives searches/recommendations/topic
    # switches (a normal new search must NEVER clear it); cleared only by RESET,
    # session expiry, an explicit "clear cart", and the successful purchase of the
    # lines it contributed to a checkout (see checkout_cart_ids above). At most one
    # CartLine per product id -- see SessionActivityStore.add_to_cart().


class SessionActivityStore:
    """In-memory per-session store, same persistence posture as
    StateManager (a handful of concurrent conversations at this project's
    scale, no external persistence layer needed)."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionActivity] = {}
        self._access_count = 0

    def get(self, session_id: str, now: float | None = None) -> SessionActivity:
        """An unseen session id starts with a completely empty/zeroed
        SessionActivity, created (and thereafter retained) on first
        access.

        Sprint 6 lifecycle task: when `now` is given and this session has
        been idle past SESSION_TTL_SECONDS, it is lazily reset first (same
        full wipe as reset()) before being returned -- an expired session
        must look exactly like a fresh one, memory included. `now` also
        stamps this access as the new last-activity time, refreshing the
        TTL. `now=None` (the default) skips both the expiry check and the
        stamp -- existing callers/tests that don't care about TTL are
        unaffected."""
        if now is not None and self.is_expired(session_id, now):
            self.reset(session_id)
        activity = self._sessions.setdefault(session_id, SessionActivity())
        if now is not None:
            activity.last_active = now
            self._access_count += 1
        return activity

    def reclaim_abandoned(self, now: float, *, exclude: str) -> list[str]:
        """Abandoned-session reclaim task: removes every OTHER session
        (never `exclude`, the one this request is actually serving) whose
        own last_active is past SESSION_TTL_SECONDS -- a session whose id
        is simply never accessed again (get()'s own lazy expiry only ever
        revisits the SAME id being accessed, so it can't reclaim this on
        its own). Amortized -- see _RECLAIM_CHECK_EVERY -- so this only
        actually scans self._sessions on the 1st real access and every
        _RECLAIM_CHECK_EVERY-th one after; every other call is a cheap
        no-op. Returns the reclaimed ids so a caller that owns OTHER
        per-session state tied to this same lifecycle (StateManager's
        shopping FilterSet) can clear its own half too -- mirrors
        is_expired()'s existing peek-then-caller-clears split."""
        if (self._access_count - 1) % _RECLAIM_CHECK_EVERY != 0:
            return []
        expired = [
            sid for sid, activity in self._sessions.items()
            if sid != exclude and activity.last_active is not None
            and (now - activity.last_active) > SESSION_TTL_SECONDS
        ]
        for sid in expired:
            del self._sessions[sid]
        return expired

    def is_expired(self, session_id: str, now: float) -> bool:
        """Peek-only (no mutation) -- lets a caller that owns OTHER
        per-session state tied to this same lifecycle (StateManager's
        shopping FilterSet) clear its own half before this store's next
        get() lazily wipes its own. False for a session never seen at all
        (nothing to expire)."""
        activity = self._sessions.get(session_id)
        if activity is None or activity.last_active is None:
            return False
        return (now - activity.last_active) > SESSION_TTL_SECONDS

    def append_history(self, session_id: str, message: str, intent: str | None = None) -> None:
        """Append one raw CURRENT user message to this session's
        continuity history, bounded to the last HISTORY_WINDOW entries --
        and, Sprint 6, to the same session's unbounded `turns` memory log
        (same call, same acceptance conditions, no separate call site to
        keep in sync).

        Callers (Part 3 orchestration) must only call this for a turn
        that: was not blocked by the System Guard, was successfully
        understood by the LLM, and resolved to an intent other than
        OFF_TOPIC. A NO_OP turn -- including a successful
        recommendation/exploration turn, and a memory-recall turn -- DOES
        get appended: it is real on-topic conversational content relevant
        to future continuity judgments (Sprint 4 design pass) and to
        future memory recall alike. `intent` is optional purely for
        pre-Sprint-6 callers/tests; omitting it just leaves that one
        TurnRecord's intent unset."""
        activity = self.get(session_id)
        activity.history.append(message)
        if len(activity.history) > HISTORY_WINDOW:
            del activity.history[: len(activity.history) - HISTORY_WINDOW]
        activity.turns.append(TurnRecord(message=message, intent=intent))

    def set_last_turn_outcome(self, session_id: str, outcome: str) -> None:
        """Compact conversation-state/LLM-memory task: attaches a short,
        deterministic outcome summary to the most recently appended
        TurnRecord (see llm_context.py's summarize_turn_outcome()) -- a
        no-op when no turn was appended this request (e.g. a blocked or
        social-fast-path turn), since `turns` is then unchanged."""
        activity = self.get(session_id)
        if activity.turns:
            activity.turns[-1].outcome = outcome

    def increment_turn_count(self, session_id: str) -> int:
        """Counts a successfully understood/accepted conversational turn
        (including OFF_TOPIC, which is accepted but not stored in
        continuity history). Callers must not call this for an oversized,
        rapid-repeat-blocked, or LLM/API-failed request."""
        activity = self.get(session_id)
        activity.turn_count += 1
        return activity.turn_count

    def reset_relaxation_step(self, session_id: str) -> None:
        """Any real canonical state-changing turn (search/add_filter/
        modify_filter/remove_filter) resets automatic price relaxation --
        the search criteria changed, so any previously widened budget no
        longer applies. Touches only relaxation_step -- see reset() for
        the full RESET-intent wipe."""
        self.get(session_id).relaxation_step = 0

    def set_pending_action(
        self, session_id: str, kind: str, shown_product_ids: list[int] | None = None,
    ) -> None:
        """Records what Spini is waiting on for this session (a
        recommendation offer, a checkout awaiting clarification, ...).
        Replaces any prior pending action outright."""
        self.get(session_id).pending_action = PendingAction(
            kind=kind, shown_product_ids=list(shown_product_ids or []),
        )

    def pop_pending_action(self, session_id: str) -> PendingAction | None:
        """Reads AND clears this session's pending_action in one step.
        Sprint 6 recommendation-offer task: a pending offer is consumed by
        the very next turn regardless of whether that turn actually
        resolves it (accept/reject) -- the caller decides what to do with
        the returned value, but the stored state is already gone either
        way, which is exactly what keeps a stale offer from surviving to
        hijack a LATER, unrelated turn."""
        activity = self.get(session_id)
        pending = activity.pending_action
        activity.pending_action = None
        return pending

    def set_last_shown_products(self, session_id: str, products: list[DisplayedProduct]) -> None:
        """Records the MOST RECENTLY displayed real, ordered product list
        (Sprint 6 product-selection task). Callers must only call this
        when something real was actually shown (a non-empty list) -- a
        turn that shows nothing new (zero-result, out-of-catalog, memory
        recall, social, off-topic) must never call this, so the OLD list
        (and therefore "first one"/"the Juhayna one" resolution) survives
        untouched across those turns."""
        activity = self.get(session_id)
        activity.last_shown_products = list(products)
        activity.recent_shown_pages.append(list(products))
        if len(activity.recent_shown_pages) > HISTORY_WINDOW:
            del activity.recent_shown_pages[: len(activity.recent_shown_pages) - HISTORY_WINDOW]

    def recent_shown_products(self, session_id: str) -> list[DisplayedProduct]:
        """Cross-turn multi-product checkout task: the bounded-recency view
        NAMED/exact selection is allowed to search (raw-message exact-title
        matches and selection_text), unlike positional/vague references
        (ordinal/last), which must use last_shown_products (the current
        page) only. Flattens recent_shown_pages newest-page-first,
        deduplicating by product id so the SAME product shown twice keeps
        only its freshest snapshot (price included) and is never offered
        twice for the same raw-message multi-select match."""
        activity = self.get(session_id)
        seen: dict[int, DisplayedProduct] = {}
        for page in reversed(activity.recent_shown_pages):
            for p in page:
                seen.setdefault(p.id, p)
        return list(seen.values())

    def clear_recent_shown_products(self, session_id: str) -> None:
        """Cross-turn multi-product checkout task: drops the bounded
        recent-pages memory -- called on RESET and completed checkout
        (mirrors clear_checkout()'s own call sites for those two cases;
        deliberately NOT called from a normal new search/topic switch,
        which must keep letting an earlier turn's named product stay
        checkout-selectable -- that survival is this whole task's point).
        Session expiry/New Chat need no explicit call: reset() below
        already replaces the whole SessionActivity, this field included."""
        self.get(session_id).recent_shown_pages = []

    def set_selected_products(self, session_id: str, lines: list[CartLine]) -> None:
        """Records the CHECKOUT SELECTION -- the exact product(s)+quantity
        being purchased NOW (cart/direct-checkout architecture task) -- the
        full small snapshot(s) (incl. price) so the checkout draft always
        has real name/price/quantity data cached, even if a subsequent
        search replaces last_shown_products in between, or the cart itself
        later changes. Replaces any prior selection outright. Never mutates
        StateManager/last_shown_products/cart."""
        self.get(session_id).selected_products = list(lines)

    def start_checkout(self, session_id: str, *, cart_ids: list[int] | None = None) -> None:
        """Sprint 6 checkout task (generalized): marks a checkout draft as
        open for the session's current selected_products. `cart_ids`
        (cart/direct-checkout architecture task): the cart product ids this
        draft was built from -- non-empty for a cart checkout (all or
        subset), left [] (the default) for a DIRECT checkout, so
        checkout_submission.py knows whether/which cart lines to remove
        after a successful purchase. Never mutates StateManager, never
        invents a product -- callers must already have confirmed a real,
        non-empty selected_products exists."""
        activity = self.get(session_id)
        activity.checkout_active = True
        activity.checkout_cart_ids = list(cart_ids or [])

    def clear_checkout(self, session_id: str) -> None:
        """Sprint 6 checkout task: closes any open checkout draft AND drops
        the selection it was for (and its cart provenance) -- called on
        RESET, a new successful normal search/topic switch, and
        post-checkout-success (checkout_submission.py) so neither a stale
        draft nor an already-decided (open OR just-purchased) selection can
        survive into an unrelated later shopping flow. Never touches
        `cart` itself -- see clear_cart()/checkout_submission.py for the
        two separate, deliberate ways cart lines actually get removed."""
        activity = self.get(session_id)
        activity.checkout_active = False
        activity.selected_products = []
        activity.checkout_cart_ids = []

    # ------------------------------------------------------------------
    # Cart/direct-checkout architecture task: the durable cart (concept B).
    # Product identity/price always come from a real DisplayedProduct
    # snapshot the caller already deterministically resolved (shown-product
    # or existing-cart authority) -- never a client/LLM value, never a
    # fresh catalog lookup. All-or-nothing multi-item mutation is the
    # CALLER's responsibility (resolve fully first, mutate only on full
    # success); these methods themselves just apply an already-decided,
    # already-valid change.
    # ------------------------------------------------------------------

    def add_to_cart(self, session_id: str, products: list[DisplayedProduct], quantity: int = 1) -> None:
        """Adds each product to the cart, or -- same product id already
        present -- increments its existing line's quantity by `quantity`
        instead of creating a second row (never a duplicate cart row for
        the same product id). Invalidates any open checkout draft first
        (see clear_checkout() call below) -- correction: a draft opened
        before this add may no longer reflect what the cart now contains."""
        self.clear_checkout(session_id)
        activity = self.get(session_id)
        by_id = {line.product.id: line for line in activity.cart}
        for product in products:
            existing = by_id.get(product.id)
            if existing is not None:
                existing.quantity += quantity
            else:
                new_line = CartLine(product=product, quantity=quantity)
                activity.cart.append(new_line)
                by_id[product.id] = new_line

    def remove_from_cart(self, session_id: str, product_ids: list[int]) -> list[CartLine]:
        """Removes every cart line whose product id is in `product_ids`.
        Returns the removed lines (for a caller's confirmation wording) --
        callers must already have resolved `product_ids` fully (all-or-
        nothing) before calling; this method itself does no resolution.
        Invalidates any open checkout draft first (correction: a removed
        product must never remain purchasable via a stale draft/form)."""
        self.clear_checkout(session_id)
        activity = self.get(session_id)
        ids = set(product_ids)
        removed = [line for line in activity.cart if line.product.id in ids]
        activity.cart = [line for line in activity.cart if line.product.id not in ids]
        return removed

    def clear_cart(self, session_id: str) -> None:
        """Empties the cart -- never touches shown-product/search state (a
        "clear cart" must not reset an unrelated active search -- see
        orchestrator.py). DOES invalidate any open checkout draft first
        (correction: a draft referencing now-cleared cart lines must never
        survive) -- the cart's own contents are otherwise untouched by
        clear_checkout()."""
        self.clear_checkout(session_id)
        self.get(session_id).cart = []

    def set_cart_quantity(self, session_id: str, product_id: int, quantity: int) -> CartLine | None:
        """Sets one existing cart line's quantity outright (never creates a
        new line -- "Product must already be in cart"). Returns the
        updated line, or None if `product_id` isn't in the cart. `quantity`
        must already be validated (>=1) by the caller. Invalidates any open
        checkout draft first (correction: a draft opened with the OLD
        quantity must never be submittable after it changes)."""
        self.clear_checkout(session_id)
        activity = self.get(session_id)
        for line in activity.cart:
            if line.product.id == product_id:
                line.quantity = quantity
                return line
        return None

    def reset(self, session_id: str) -> None:
        """Full runtime reset: history, turns (conversational memory),
        repeat bookkeeping, turn_count, relaxation_step, and last_active
        are all cleared together -- the next message in this session must
        behave exactly like a fresh, unseen session.

        Sprint 6 lifecycle task: a normal shopping "start over" (RESET
        intent) deliberately does NOT call this anymore -- it only clears
        StateManager's canonical FilterSet and this store's
        relaxation_step (see orchestrator.py's RESET branch), preserving
        conversation memory. This full wipe now fires only for genuine
        session expiration (get()/is_expired(), SESSION_TTL_SECONDS) --
        the one case where BOTH shopping state and conversational memory
        must actually disappear -- plus direct test use."""
        self._sessions[session_id] = SessionActivity()
