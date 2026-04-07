"""JustTCG API client for per-condition Pokemon card pricing.

API docs: https://justtcg.com/docs
Base URL: https://api.justtcg.com/v1

Key endpoints:
  GET  /cards?tcgplayerId=<id>  — lookup by TCGPlayer product ID (our tcg_product_id)
  POST /cards                   — batch lookup (up to 20 on free tier)
  GET  /games                   — list supported games
  GET  /sets?game=<id>          — list sets for a game

Auth: x-api-key header
Rate limits (free tier): 1,000/month, 100/day, 10/minute

Card response includes variants array with per-condition pricing:
  conditions: S (Sealed), NM (Near Mint), LP (Lightly Played),
              MP (Moderately Played), HP (Heavily Played), DMG (Damaged)
  printings:  Normal, Holofoil, Reverse Holofoil, 1st Edition Holofoil, etc.

Signup (free, no credit card): https://justtcg.com/auth/signup
Dashboard/API key: https://justtcg.com/dashboard
"""

import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_URL = "https://api.justtcg.com/v1"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DB_PATH = DATA_DIR / "justtcg_prices.db"

# Set via env: export JUSTTCG_API_KEY=tcg_...
API_KEY = os.environ.get("JUSTTCG_API_KEY", "")

# Rate limiting: free tier = 10 req/min
MIN_REQUEST_INTERVAL = 6.5  # seconds between requests (conservative)

# Consecutive-failure cap: if this many requests in a row hit 429 or other
# errors with zero successes between, abort the run.  Prevents the "zombie
# scraper stuck in retry loop forever" failure mode.  At 60s sleep per 429
# this gives ~30 minutes of stuck-retries before bailing.
MAX_CONSECUTIVE_FAILURES = 30


class ConsecutiveFailureLimit(Exception):
    """Raised when too many consecutive request failures occur in a row."""


# ---------------------------------------------------------------------------
# SQLite setup
# ---------------------------------------------------------------------------

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS justtcg_prices (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    game            TEXT NOT NULL DEFAULT 'pokemon',
    tcg_product_id  INTEGER NOT NULL,
    card_name       TEXT,
    set_name        TEXT,
    card_number     TEXT,
    rarity          TEXT,
    condition       TEXT NOT NULL,
    printing        TEXT NOT NULL,
    price           REAL,
    avg_price       REAL,
    price_change_24hr REAL,
    price_change_7d   REAL,
    last_updated    INTEGER,
    fetched_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_justtcg_product_condition
    ON justtcg_prices (tcg_product_id, condition);
CREATE INDEX IF NOT EXISTS idx_justtcg_fetched
    ON justtcg_prices (fetched_at);
CREATE INDEX IF NOT EXISTS idx_justtcg_game_product
    ON justtcg_prices (game, tcg_product_id);
"""


def _ensure_game_column(conn: sqlite3.Connection) -> None:
    """Add game column to pre-existing tables (backwards compat migration)."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(justtcg_prices)").fetchall()}
    if "game" not in cols:
        conn.execute(
            "ALTER TABLE justtcg_prices ADD COLUMN game TEXT NOT NULL DEFAULT 'pokemon'"
        )
        conn.commit()


def get_db(db_path: Path | None = None) -> sqlite3.Connection:
    """Open (and initialize) the SQLite database."""
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(CREATE_TABLE_SQL)
    _ensure_game_column(conn)
    conn.executescript(CREATE_INDEX_SQL)
    return conn


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------

class JustTCGClient:
    """Rate-limited client for the JustTCG API."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or API_KEY
        if not self.api_key:
            raise ValueError(
                "No JustTCG API key. Set JUSTTCG_API_KEY env var or pass api_key. "
                "Get a free key at https://justtcg.com/auth/signup"
            )
        self.session = requests.Session()
        self.session.headers.update({
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        })
        self._last_request_time = 0.0
        # Track quota from response metadata
        self.requests_remaining = None
        self.daily_remaining = None
        # Consecutive-failure tracking — resets to 0 on each successful request
        self._consecutive_failures = 0

    def _rate_limit(self):
        """Sleep if needed to respect rate limits."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < MIN_REQUEST_INTERVAL:
            wait = MIN_REQUEST_INTERVAL - elapsed
            logger.debug(f"Rate limiting: sleeping {wait:.1f}s")
            time.sleep(wait)
        self._last_request_time = time.monotonic()

    def _update_quota(self, meta: dict):
        """Update quota tracking from response metadata."""
        if meta:
            self.requests_remaining = meta.get("apiRequestsRemaining")
            self.daily_remaining = meta.get("apiDailyRequestsRemaining")
            if self.requests_remaining is not None:
                logger.info(
                    f"JustTCG quota: {self.requests_remaining} monthly, "
                    f"{self.daily_remaining} daily remaining"
                )

    def _note_failure(self, reason: str):
        """Increment consecutive-failure counter; abort if cap reached."""
        self._consecutive_failures += 1
        if self._consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            raise ConsecutiveFailureLimit(
                f"Aborting JustTCG run after {self._consecutive_failures} "
                f"consecutive failures (last: {reason}). API likely down or "
                f"quota exhausted — try again later."
            )

    def _note_success(self):
        """Reset consecutive-failure counter on a successful request."""
        self._consecutive_failures = 0

    def _request(self, method: str, endpoint: str, **kwargs) -> dict:
        """Make a rate-limited API request.

        Retries 429s with a 60s sleep, but tracks consecutive failures.
        After MAX_CONSECUTIVE_FAILURES (30) in a row, raises
        ConsecutiveFailureLimit so the caller can exit cleanly instead of
        spinning forever in rate-limit hell.
        """
        self._rate_limit()
        url = f"{BASE_URL}{endpoint}"
        try:
            resp = self.session.request(method, url, timeout=30, **kwargs)
        except requests.RequestException as e:
            logger.error(f"JustTCG request failed: {e}")
            self._note_failure(f"network error: {e}")
            raise

        if resp.status_code == 429:
            logger.warning(
                "JustTCG rate limit hit, waiting 60s (consecutive failures: %d/%d)",
                self._consecutive_failures + 1, MAX_CONSECUTIVE_FAILURES,
            )
            self._note_failure("429 rate limit")
            time.sleep(60)
            return self._request(method, endpoint, **kwargs)

        if resp.status_code == 401:
            raise ValueError(f"JustTCG auth failed: {resp.text}")

        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            self._note_failure(f"HTTP {resp.status_code}")
            raise

        data = resp.json()
        self._note_success()
        self._update_quota(data.get("_metadata", {}))
        return data

    # -----------------------------------------------------------------------
    # Endpoints
    # -----------------------------------------------------------------------

    def get_games(self) -> list[dict]:
        """List all supported games."""
        result = self._request("GET", "/games")
        return result.get("data", [])

    def get_sets(self, game_id: str, query: str = "") -> list[dict]:
        """List sets for a game, optionally filtered by name."""
        params = {"game": game_id}
        if query:
            params["q"] = query
        result = self._request("GET", "/sets", params=params)
        return result.get("data", [])

    def get_card_by_tcgplayer_id(
        self,
        tcgplayer_id: int,
        include_history: bool = False,
        history_duration: str = "7d",
        game: str = "pokemon",
    ) -> dict | None:
        """Look up a single card by TCGPlayer product ID.

        Returns the full card object with variants (per-condition pricing),
        or None if not found.
        """
        params = {
            "tcgplayerId": str(tcgplayer_id),
            "game": game,
            "include_price_history": str(include_history).lower(),
            "include_null_prices": "false",
        }
        if include_history:
            params["priceHistoryDuration"] = history_duration
        try:
            result = self._request("GET", "/cards", params=params)
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                return None
            raise
        cards = result.get("data", [])
        return cards[0] if cards else None

    def get_cards_batch(
        self,
        tcgplayer_ids: list[int],
        include_history: bool = False,
        game: str = "pokemon",
    ) -> list[dict]:
        """Batch lookup by TCGPlayer product IDs (max 20 on free tier).

        Returns list of card objects.
        """
        if len(tcgplayer_ids) > 20:
            raise ValueError("Batch limit is 20 cards on free tier")

        body = [{"tcgplayerId": str(tid)} for tid in tcgplayer_ids]
        params = {
            "game": game,
            "include_price_history": str(include_history).lower(),
            "include_null_prices": "false",
        }
        result = self._request("POST", "/cards", json=body, params=params)
        return result.get("data", [])

    # -----------------------------------------------------------------------
    # Storage
    # -----------------------------------------------------------------------

    def fetch_and_store(
        self,
        tcgplayer_id: int,
        db: sqlite3.Connection | None = None,
        game: str = "pokemon",
    ) -> list[dict]:
        """Fetch pricing for a card and store in SQLite. Returns variant rows."""
        card = self.get_card_by_tcgplayer_id(tcgplayer_id, game=game)
        if not card:
            logger.warning(f"Card not found for tcgplayerId={tcgplayer_id}")
            return []

        own_db = db is None
        if own_db:
            db = get_db()

        rows = []
        now = datetime.now(timezone.utc).isoformat()
        for v in card.get("variants", []):
            row = {
                "game": game,
                "tcg_product_id": tcgplayer_id,
                "card_name": card.get("name"),
                "set_name": card.get("set_name"),
                "card_number": card.get("number"),
                "rarity": card.get("rarity"),
                "condition": v.get("condition", ""),
                "printing": v.get("printing", ""),
                "price": v.get("price"),
                "avg_price": v.get("avgPrice"),
                "price_change_24hr": v.get("priceChange24hr"),
                "price_change_7d": v.get("priceChange7d"),
                "last_updated": v.get("lastUpdated"),
                "fetched_at": now,
            }
            db.execute(
                """INSERT INTO justtcg_prices
                   (game, tcg_product_id, card_name, set_name, card_number, rarity,
                    condition, printing, price, avg_price,
                    price_change_24hr, price_change_7d, last_updated, fetched_at)
                   VALUES (:game, :tcg_product_id, :card_name, :set_name, :card_number,
                           :rarity, :condition, :printing, :price, :avg_price,
                           :price_change_24hr, :price_change_7d, :last_updated,
                           :fetched_at)""",
                row,
            )
            rows.append(row)

        db.commit()
        if own_db:
            db.close()

        logger.info(
            f"Stored {len(rows)} variants for {card.get('name')} "
            f"(tcgplayerId={tcgplayer_id})"
        )
        return rows

    def fetch_and_store_batch(
        self,
        tcgplayer_ids: list[int],
        db: sqlite3.Connection | None = None,
        game: str = "pokemon",
    ) -> int:
        """Batch fetch and store. Returns total variant rows stored."""
        cards = self.get_cards_batch(tcgplayer_ids, game=game)

        own_db = db is None
        if own_db:
            db = get_db()

        total = 0
        now = datetime.now(timezone.utc).isoformat()
        for card in cards:
            tid = card.get("tcgplayerId")
            for v in card.get("variants", []):
                db.execute(
                    """INSERT INTO justtcg_prices
                       (game, tcg_product_id, card_name, set_name, card_number, rarity,
                        condition, printing, price, avg_price,
                        price_change_24hr, price_change_7d, last_updated, fetched_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        game, tid, card.get("name"), card.get("set_name"),
                        card.get("number"), card.get("rarity"),
                        v.get("condition", ""), v.get("printing", ""),
                        v.get("price"), v.get("avgPrice"),
                        v.get("priceChange24hr"), v.get("priceChange7d"),
                        v.get("lastUpdated"), now,
                    ),
                )
                total += 1

        db.commit()
        if own_db:
            db.close()

        logger.info(f"Batch stored {total} variants for {len(cards)} cards")
        return total


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def get_latest_prices(
    tcg_product_id: int,
    db: sqlite3.Connection | None = None,
) -> list[dict]:
    """Get the most recent pricing data for a card from local SQLite."""
    own_db = db is None
    if own_db:
        db = get_db()

    rows = db.execute(
        """SELECT * FROM justtcg_prices
           WHERE tcg_product_id = ?
           ORDER BY fetched_at DESC
           LIMIT 20""",
        (tcg_product_id,),
    ).fetchall()

    if own_db:
        db.close()
    return [dict(r) for r in rows]


def print_price_breakdown(rows: list[dict]):
    """Pretty-print per-condition pricing for a card."""
    if not rows:
        print("  No pricing data available")
        return

    card_name = rows[0].get("card_name", "Unknown")
    set_name = rows[0].get("set_name", "Unknown")
    print(f"\n  {card_name} — {set_name}")
    print(f"  {'Printing':<20} {'Condition':<15} {'Price':>8} {'Avg':>8} {'7d Chg':>8}")
    print(f"  {'-'*20} {'-'*15} {'-'*8} {'-'*8} {'-'*8}")

    for r in sorted(rows, key=lambda x: (x.get("printing", ""), x.get("condition", ""))):
        price = f"${r['price']:.2f}" if r.get("price") is not None else "N/A"
        avg = f"${r['avg_price']:.2f}" if r.get("avg_price") is not None else "N/A"
        chg = f"{r['price_change_7d']:+.1f}%" if r.get("price_change_7d") is not None else "N/A"
        print(f"  {r.get('printing', ''):<20} {r.get('condition', ''):<15} {price:>8} {avg:>8} {chg:>8}")


# ---------------------------------------------------------------------------
# Test / demo
# ---------------------------------------------------------------------------

# Well-known TCGPlayer product IDs for popular Pokemon cards:
DEMO_CARDS = {
    "Charizard (Base Set Holo)": 95317,
    "Pikachu (Base Set)": 89356,
    "Mewtwo (Base Set Holo)": 89608,
    "Lugia (Neo Genesis Holo)": 88591,
}


def test_api():
    """Test the JustTCG API with popular cards.

    Requires JUSTTCG_API_KEY env var to be set.
    Get a free key at https://justtcg.com/auth/signup (no credit card).
    """
    print("=" * 70)
    print("JustTCG API Test — Per-Condition Pokemon Card Pricing")
    print("=" * 70)

    if not API_KEY:
        print(
            "\nNo API key found. To use this module:\n"
            "  1. Sign up free at https://justtcg.com/auth/signup\n"
            "  2. Get your API key from https://justtcg.com/dashboard\n"
            "  3. export JUSTTCG_API_KEY=tcg_your_key_here\n"
            "  4. Run again: python -m cardprice.scrapers.justtcg_prices\n"
        )
        # Still initialize the DB schema
        db = get_db()
        db.close()
        print(f"SQLite database initialized at: {DB_PATH}")
        return

    client = JustTCGClient()
    db = get_db()

    print(f"\nUsing API key: {client.api_key[:8]}...{client.api_key[-4:]}")
    print(f"SQLite DB: {DB_PATH}\n")

    # Test single lookups
    for name, tcg_id in DEMO_CARDS.items():
        print(f"\nLooking up {name} (tcgplayerId={tcg_id})...")
        try:
            rows = client.fetch_and_store(tcg_id, db=db)
            print_price_breakdown(rows)
        except Exception as e:
            print(f"  Error: {e}")

    # Show quota status
    print(f"\n{'=' * 70}")
    print(f"Quota remaining — Monthly: {client.requests_remaining}, "
          f"Daily: {client.daily_remaining}")

    # Test batch lookup
    print(f"\n{'=' * 70}")
    print("Testing batch lookup...")
    try:
        batch_ids = list(DEMO_CARDS.values())
        total = client.fetch_and_store_batch(batch_ids, db=db)
        print(f"  Batch stored {total} variant rows for {len(batch_ids)} cards")
    except Exception as e:
        print(f"  Batch error: {e}")

    db.close()
    print(f"\nDone. Data stored in {DB_PATH}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_api()
