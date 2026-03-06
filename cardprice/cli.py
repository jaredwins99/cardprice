"""CLI entry point for the cardprice weekly refresh pipeline.

Usage:
    python -m cardprice.cli refresh       # weekly refresh (live prices + card catalog)
    python -m cardprice.cli backfill      # historical backfill
    python -m cardprice.cli migrate       # run DB migrations
    python -m cardprice.cli mapping       # TCGCSV product-to-card mapping
    python -m cardprice.cli status        # show DB row counts and date ranges
    python -m cardprice.cli pokeapi        # fetch Pokemon species metadata from PokeAPI
    python -m cardprice.cli download-images # download card images from pokemontcg.io
    python -m cardprice.cli priority      # build card scrape priority queue
    python -m cardprice.cli smogon        # fetch Smogon competitive usage stats
    python -m cardprice.cli scan           # scan card image to identify
    python -m cardprice.cli inventory      # manage card inventory (list/add/remove/value/export)
    python -m cardprice.cli valuation      # snapshot collection value
    python -m cardprice.cli watch          # watch folder for card images to auto-scan
    python -m cardprice.cli server         # start card scanner HTTP server
    python -m cardprice.cli telegram       # start Telegram card scanner bot
    python -m cardprice.cli build-hash-index  # build perceptual hash database
    python -m cardprice.cli build-dino-index  # build DINOv2 FAISS index
    python -m cardprice.cli build-clip-index  # build CLIP embedding indexes
"""

import argparse
import csv
import io
import json
import logging
import os
import sys
from datetime import date
from decimal import Decimal

from sqlalchemy import text

from cardprice.db.session import SessionLocal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# ---------------------------------------------------------------------------
# Condition multipliers for valuation
# Updated March 2026 from market research (see condition_pricing.py docstring)
# ---------------------------------------------------------------------------
CONDITION_MULTIPLIERS = {
    "PSA10": Decimal("3.5"),
    "PSA9": Decimal("1.8"),
    "PSA8": Decimal("1.3"),
    "NM": Decimal("1.0"),
    "LP": Decimal("0.80"),
    "MP": Decimal("0.55"),
    "HP": Decimal("0.30"),
    "DMG": Decimal("0.12"),
}


def _effective_condition(row):
    """Return the valuation key for a user_inventory row.

    If the card has a grade (e.g. PSA 10), return 'PSA10' / 'PSA9'.
    Otherwise fall back to the condition column (NM, LP, ...).
    """
    if row.grade_authority and row.grade:
        key = f"{row.grade_authority}{row.grade}".upper()
        if key in CONDITION_MULTIPLIERS:
            return key
    return (row.condition or "NM").upper()


# ---------------------------------------------------------------------------
# Existing pipeline commands
# ---------------------------------------------------------------------------


def cmd_refresh(args):
    """Run weekly refresh: live prices + card catalog update."""
    from cardprice.scrapers.tcgcsv import ingest_live_prices
    from cardprice.scrapers.pokemontcg import ingest_all

    with SessionLocal() as session:
        print("=== Card catalog update (pokemontcg.io) ===")
        ingest_all(session)

        print("\n=== Live price ingestion (TCGCSV) ===")
        summary = ingest_live_prices(session)
        print(f"  Date:     {summary['date']}")
        print(f"  Groups:   {summary['groups']}")
        print(f"  Rows:     {summary['total_price_rows']}")
        print(f"  Inserted: {summary['inserted']}")
        print(f"  Errors:   {summary['errors']}")

    print("\nRefresh complete.")


def cmd_backfill(args):
    """Run historical price backfill from TCGCSV archives."""
    from cardprice.scrapers.backfill import backfill

    start = date.fromisoformat(args.start) if args.start else None
    end = date.fromisoformat(args.end) if args.end else None

    summary = backfill(start=start, end=end)
    print(f"\nBackfill summary:")
    print(f"  Days attempted: {summary['days_attempted']}")
    print(f"  Days done:      {summary['days_done']}")
    print(f"  Total rows:     {summary['total_rows']}")
    print(f"  Inserted:       {summary['total_inserted']}")
    if summary["errors"]:
        print(f"  Failed dates:   {', '.join(summary['errors'])}")


def cmd_migrate(args):
    """Run DB migrations (create all tables)."""
    from cardprice.db.migrate import run

    run()


def cmd_mapping(args):
    """Run TCGCSV product-to-card mapping."""
    from cardprice.scrapers.mapping import run_mapping

    with SessionLocal() as session:
        run_mapping(session)


def cmd_pokeapi(args):
    """Fetch Pokemon species metadata from PokeAPI."""
    from cardprice.scrapers.pokeapi import fetch_all_species

    with SessionLocal() as session:
        print("=== Fetching Pokemon species metadata from PokeAPI ===")
        summary = fetch_all_species(session)
        print(f"\nPokeAPI complete:")
        print(f"  Total:    {summary['total']:,}")
        print(f"  Inserted: {summary['inserted']:,}")
        print(f"  Errors:   {summary['errors']:,}")


def cmd_download_images(args):
    """Download card images from pokemontcg.io URLs in dim_cards."""
    from cardprice.scrapers.image_downloader import download_card_images

    with SessionLocal() as session:
        print(f"=== Downloading {args.size} card images to {args.output} ===")
        stats = download_card_images(
            session,
            output_dir=args.output,
            size=args.size,
            batch_size=args.batch_size,
        )
        print(f"\nDownload complete:")
        print(f"  Downloaded: {stats['downloaded']:,}")
        print(f"  Skipped:    {stats['skipped']:,}")
        print(f"  Failed:     {stats['failed']:,}")


def cmd_priority(args):
    """Build card scrape priority queue."""
    from cardprice.scrapers.priority import build_priority_queue

    with SessionLocal() as session:
        print(f"=== Building scrape priority queue (top_n={args.top_n}) ===")
        summary = build_priority_queue(session, top_n=args.top_n)
        print(f"\nPriority queue built:")
        print(f"  Cards scored: {summary['cards_scored']:,}")
        if summary.get("min_score") is not None:
            print(f"  Score range:  {summary['min_score']:.4f} - {summary['max_score']:.4f}")
            print(f"  Avg score:    {summary['avg_score']:.4f}")


def cmd_smogon(args):
    """Fetch Smogon competitive usage stats."""
    from cardprice.scrapers.smogon import fetch_smogon_usage

    formats = args.formats.split(",") if args.formats else None

    with SessionLocal() as session:
        print("=== Fetching Smogon competitive usage stats ===")
        summary = fetch_smogon_usage(session, formats=formats)
        print(f"\nSmogon fetch complete:")
        print(f"  Formats fetched: {summary['formats_fetched']}")
        print(f"  Total rows:      {summary['total_rows']:,}")
        if summary["errors"]:
            print(f"  Failed formats:  {', '.join(summary['errors'])}")


def cmd_status(args):
    """Show DB row counts and date ranges."""
    queries = {
        "dim_pokemon": "SELECT COUNT(*) FROM dim_pokemon",
        "dim_sets": "SELECT COUNT(*) FROM dim_sets",
        "dim_cards": "SELECT COUNT(*) FROM dim_cards",
        "dim_cards (mapped)": "SELECT COUNT(*) FROM dim_cards WHERE tcg_product_id IS NOT NULL",
        "dim_marketplaces": "SELECT COUNT(*) FROM dim_marketplaces",
        "fact_market_prices": "SELECT COUNT(*) FROM fact_market_prices",
        "fact_sales": "SELECT COUNT(*) FROM fact_sales",
        "user_inventory": "SELECT COUNT(*) FROM user_inventory",
        "inventory_scans": "SELECT COUNT(*) FROM inventory_scans",
        "inventory_valuations": "SELECT COUNT(*) FROM inventory_valuations",
    }

    date_queries = {
        "fact_market_prices": (
            "SELECT MIN(price_date), MAX(price_date) FROM fact_market_prices"
        ),
        "fact_sales": (
            "SELECT MIN(sale_date)::date, MAX(sale_date)::date FROM fact_sales"
        ),
    }

    with SessionLocal() as session:
        print("=== Row Counts ===")
        for label, sql in queries.items():
            try:
                count = session.execute(text(sql)).scalar()
                print(f"  {label:30s} {count:>12,}")
            except Exception as e:
                print(f"  {label:30s} ERROR: {e}")
                session.rollback()

        print("\n=== Date Ranges ===")
        for label, sql in date_queries.items():
            try:
                row = session.execute(text(sql)).fetchone()
                if row and row[0]:
                    print(f"  {label:30s} {row[0]}  to  {row[1]}")
                else:
                    print(f"  {label:30s} (no data)")
            except Exception as e:
                print(f"  {label:30s} ERROR: {e}")
                session.rollback()


# ---------------------------------------------------------------------------
# scan -- identify a card from an image (helpers + command)
# ---------------------------------------------------------------------------


def _resolve_image_paths(args):
    """Resolve image paths from CLI args (single file or directory).

    Returns a list of absolute paths or calls sys.exit on error.
    """
    image_paths = []
    if args.dir:
        dirpath = os.path.abspath(args.dir)
        if not os.path.isdir(dirpath):
            print(f"ERROR: directory not found: {dirpath}")
            sys.exit(1)
        for fname in sorted(os.listdir(dirpath)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                image_paths.append(os.path.join(dirpath, fname))
        if not image_paths:
            print(f"No image files found in {dirpath}")
            sys.exit(1)
    elif args.image_path:
        p = os.path.abspath(args.image_path)
        if not os.path.isfile(p):
            print(f"ERROR: file not found: {p}")
            sys.exit(1)
        image_paths.append(p)
    else:
        print("ERROR: provide <image_path> or --dir <directory>")
        sys.exit(1)
    return image_paths


def _run_model(model, img_path, session):
    """Run a card identification model on an image.

    Returns (card_id, confidence, raw_response).  On error, prints a
    message and returns (None, 0.0, {}).
    """
    card_id = None
    confidence = 0.0
    raw_response = {}

    try:
        if model == "claude-haiku-4-5":
            from cardprice.ml.claude_scanner import scan_card, match_to_database
            result = scan_card(img_path, model="claude-haiku-4-5")
            matched_id, match_conf = match_to_database(result, session)
            card_id = matched_id
            confidence = match_conf if matched_id else result.get("confidence", 0.0)
            raw_response = result
        elif model == "clip":
            from cardprice.ml.clip_matcher import identify_card as clip_identify
            matches = clip_identify(img_path)
            if matches:
                card_id, confidence = matches[0]
                raw_response = {"top_matches": matches}
        elif model == "dino":
            from cardprice.ml.dino_matcher import identify_card as dino_identify
            matches = dino_identify(img_path)
            if matches:
                card_id, confidence = matches[0]
                raw_response = {"top_matches": matches}
        elif model == "hash":
            from cardprice.ml.hash_matcher import match_card, classify_match
            matches = match_card(img_path)
            if matches:
                raw_card_id = matches[0][0]
                # Convert hash DB key format (base1-4_normal) to
                # dim_cards card_id format (base1-4/normal) if needed
                if "/" not in raw_card_id and "_" in raw_card_id:
                    last_us = raw_card_id.rfind("_")
                    card_id = raw_card_id[:last_us] + "/" + raw_card_id[last_us + 1:]
                else:
                    card_id = raw_card_id
                distance = int(matches[0][1])
                confidence = max(0.0, 1.0 - distance / 15.0)
                raw_response = {
                    "matches": [[str(m[0]), int(m[1])] for m in matches[:20]],
                    "confidence_label": classify_match(distance),
                }
        elif model == "cascade":
            from cardprice.ml import identify_card as cascade_identify
            result = cascade_identify(img_path, session)
            card_id = result["card_id"]
            confidence = result["confidence"]
            raw_response = result["raw_response"]
            if result["method"]:
                print(f"  Method: {result['method']}")
        else:
            print(f"  ERROR: unknown model '{model}'")
    except Exception as e:
        print(f"  ERROR ({model}): {e}")

    return card_id, confidence, raw_response


def cmd_scan(args):
    """Scan a card image (or directory of images) to identify it."""
    image_paths = _resolve_image_paths(args)
    model = args.model
    auto_threshold = 0.9

    total = len(image_paths)
    identified = 0
    added = 0

    with SessionLocal() as session:
        for img_path in image_paths:
            print(f"\n=== Scanning: {img_path} ===")
            print(f"  Model: {model}")

            card_id, confidence, raw_response = _run_model(model, img_path, session)

            # Record the scan attempt
            session.execute(
                text("""
                    INSERT INTO inventory_scans
                        (image_path, identified_card_id, confidence,
                         model_used, raw_response, accepted)
                    VALUES (:path, :card_id, :conf, :model, :raw, :accepted)
                """),
                {
                    "path": img_path,
                    "card_id": card_id,
                    "conf": float(confidence),
                    "model": model,
                    "raw": json.dumps(raw_response, default=str),
                    "accepted": bool(confidence >= auto_threshold) if card_id else False,
                },
            )
            session.commit()

            if card_id:
                identified += 1
                row = session.execute(
                    text("SELECT name, set_id, variant FROM dim_cards WHERE card_id = :cid"),
                    {"cid": card_id},
                ).fetchone()
                name_str = (
                    f"{row.name} ({row.set_id}, {row.variant})" if row else card_id
                )
                print(f"  Identified: {name_str}")
                print(f"  Confidence: {confidence:.2%}")
                if confidence >= auto_threshold:
                    print("  Auto-accepted (confidence >= 90%)")
                else:
                    print("  Below auto-accept threshold; review recommended")

                if args.add and confidence >= args.threshold:
                    session.execute(text("""
                        INSERT INTO user_inventory (card_id, quantity, condition, notes)
                        VALUES (:cid, 1, 'NM', :notes)
                    """), {"cid": card_id, "notes": f"Scanned via {model} (conf={confidence:.2f})"})
                    session.commit()
                    added += 1
                    print("  Added to inventory!")
            else:
                print("  No card identified.")

    print(f"\nBatch summary: {identified}/{total} identified, {added}/{total} added to inventory")


# ---------------------------------------------------------------------------
# inventory -- manage card inventory (list / add / remove / value / export)
# ---------------------------------------------------------------------------


def cmd_inventory(args):
    """Dispatch inventory sub-subcommands."""
    inv_action = args.inv_action
    if not inv_action:
        print("Usage: cardprice inventory {list,add,remove,value,export}")
        sys.exit(1)

    handler = {
        "list": _inv_list,
        "add": _inv_add,
        "remove": _inv_remove,
        "value": _inv_value,
        "export": _inv_export,
    }
    handler[inv_action](args)


def _inv_list(args):
    """Show all cards in inventory with current market values."""
    with SessionLocal() as session:
        rows = session.execute(
            text("""
                SELECT
                    ui.id,
                    ui.card_id,
                    dc.name AS card_name,
                    dc.set_id,
                    ui.quantity,
                    ui.condition,
                    ui.grade_authority,
                    ui.grade,
                    ui.acquisition_price,
                    ui.acquisition_date,
                    ui.notes,
                    lp.market_price
                FROM user_inventory ui
                JOIN dim_cards dc ON dc.card_id = ui.card_id
                LEFT JOIN LATERAL (
                    SELECT market_price
                    FROM fact_market_prices fmp
                    WHERE fmp.card_id = ui.card_id
                    ORDER BY fmp.price_date DESC
                    LIMIT 1
                ) lp ON true
                ORDER BY dc.name, ui.condition
            """)
        ).fetchall()

        if not rows:
            print("Inventory is empty.")
            return

        print(
            f"{'ID':>5}  {'Card':40s}  {'Set':12s}  {'Qty':>3}  {'Cond':4s}  "
            f"{'Grade':8s}  {'Paid':>8s}  {'Market':>8s}"
        )
        print("-" * 100)
        for r in rows:
            grade_str = (
                f"{r.grade_authority}{r.grade}" if r.grade_authority else ""
            )
            paid = f"${r.acquisition_price:.2f}" if r.acquisition_price else "-"
            mkt = f"${r.market_price:.2f}" if r.market_price else "-"
            print(
                f"{r.id:>5}  {r.card_name:40s}  {r.set_id:12s}  {r.quantity:>3}  "
                f"{r.condition or '-':4s}  {grade_str:8s}  {paid:>8s}  {mkt:>8s}"
            )

        print(f"\nTotal items: {len(rows)}")


def _inv_add(args):
    """Manually add a card to inventory."""
    card_id = args.card_id
    condition = args.condition.upper()
    price = Decimal(str(args.price)) if args.price is not None else None
    quantity = args.quantity
    notes = args.notes

    if condition not in ("NM", "LP", "MP", "HP", "DMG"):
        print(f"ERROR: invalid condition '{condition}'. Must be NM/LP/MP/HP/DMG.")
        sys.exit(1)

    with SessionLocal() as session:
        # Verify card exists
        exists = session.execute(
            text("SELECT 1 FROM dim_cards WHERE card_id = :cid"),
            {"cid": card_id},
        ).fetchone()
        if not exists:
            print(f"ERROR: card_id '{card_id}' not found in dim_cards.")
            sys.exit(1)

        session.execute(
            text("""
                INSERT INTO user_inventory
                    (card_id, quantity, condition, acquisition_price,
                     acquisition_date, notes)
                VALUES (:cid, :qty, :cond, :price, :adate, :notes)
            """),
            {
                "cid": card_id,
                "qty": quantity,
                "cond": condition,
                "price": price,
                "adate": date.today(),
                "notes": notes,
            },
        )
        session.commit()

        # Fetch card name for confirmation
        row = session.execute(
            text("SELECT name FROM dim_cards WHERE card_id = :cid"),
            {"cid": card_id},
        ).fetchone()
        price_str = f" @ ${price:.2f}" if price else ""
        print(f"Added {quantity}x {row.name} ({card_id}) [{condition}]{price_str}")


def _inv_remove(args):
    """Remove a card from inventory by inventory row ID."""
    inv_id = args.id

    with SessionLocal() as session:
        row = session.execute(
            text("""
                SELECT ui.id, dc.name, ui.card_id, ui.condition
                FROM user_inventory ui
                JOIN dim_cards dc ON dc.card_id = ui.card_id
                WHERE ui.id = :iid
            """),
            {"iid": inv_id},
        ).fetchone()

        if not row:
            print(f"ERROR: inventory item #{inv_id} not found.")
            sys.exit(1)

        session.execute(
            text("DELETE FROM user_inventory WHERE id = :iid"),
            {"iid": inv_id},
        )
        session.commit()
        print(f"Removed #{inv_id}: {row.name} ({row.card_id}) [{row.condition}]")


def _inv_value(args):
    """Show total collection value using latest prices + condition multipliers."""
    with SessionLocal() as session:
        rows = session.execute(
            text("""
                SELECT
                    ui.id,
                    ui.card_id,
                    dc.name AS card_name,
                    ui.quantity,
                    ui.condition,
                    ui.grade_authority,
                    ui.grade,
                    lp.market_price
                FROM user_inventory ui
                JOIN dim_cards dc ON dc.card_id = ui.card_id
                LEFT JOIN LATERAL (
                    SELECT market_price
                    FROM fact_market_prices fmp
                    WHERE fmp.card_id = ui.card_id
                    ORDER BY fmp.price_date DESC
                    LIMIT 1
                ) lp ON true
                ORDER BY dc.name
            """)
        ).fetchall()

        if not rows:
            print("Inventory is empty.")
            return

        total_value = Decimal("0")
        total_cards = 0
        unpriced = 0

        print(
            f"{'Card':40s}  {'Cond':6s}  {'Qty':>3}  {'Market':>8s}  "
            f"{'Mult':>5s}  {'Value':>10s}"
        )
        print("-" * 85)

        for r in rows:
            cond_key = _effective_condition(r)
            mult = CONDITION_MULTIPLIERS.get(cond_key, Decimal("1.0"))
            qty = r.quantity or 1
            total_cards += qty

            if r.market_price:
                base = Decimal(str(r.market_price))
                value = base * mult * qty
                total_value += value
                print(
                    f"{r.card_name:40s}  {cond_key:6s}  {qty:>3}  "
                    f"${base:>7.2f}  {mult:>5.2f}  ${value:>9.2f}"
                )
            else:
                unpriced += 1
                print(
                    f"{r.card_name:40s}  {cond_key:6s}  {qty:>3}  "
                    f"{'N/A':>8s}  {mult:>5.2f}  {'N/A':>10s}"
                )

        print("-" * 85)
        print(f"  Total cards:  {total_cards:,}")
        print(f"  Total value:  ${total_value:,.2f}")
        if unpriced:
            print(f"  Unpriced:     {unpriced} items (no market data)")


def _inv_export(args):
    """Export inventory to CSV."""
    fmt = args.format
    if fmt != "csv":
        print(f"ERROR: unsupported export format '{fmt}'. Only 'csv' is supported.")
        sys.exit(1)

    with SessionLocal() as session:
        rows = session.execute(
            text("""
                SELECT
                    ui.id,
                    ui.card_id,
                    dc.name AS card_name,
                    dc.set_id,
                    dc.rarity,
                    ui.quantity,
                    ui.condition,
                    ui.grade_authority,
                    ui.grade,
                    ui.acquisition_price,
                    ui.acquisition_date,
                    ui.notes,
                    lp.market_price,
                    lp.price_date AS latest_price_date
                FROM user_inventory ui
                JOIN dim_cards dc ON dc.card_id = ui.card_id
                LEFT JOIN LATERAL (
                    SELECT market_price, price_date
                    FROM fact_market_prices fmp
                    WHERE fmp.card_id = ui.card_id
                    ORDER BY fmp.price_date DESC
                    LIMIT 1
                ) lp ON true
                ORDER BY dc.name
            """)
        ).fetchall()

        if not rows:
            print("Inventory is empty, nothing to export.")
            return

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "inv_id", "card_id", "card_name", "set_id", "rarity",
            "quantity", "condition", "grade_authority", "grade",
            "acquisition_price", "acquisition_date", "notes",
            "market_price", "latest_price_date",
            "condition_mult", "adjusted_value",
        ])

        for r in rows:
            cond_key = _effective_condition(r)
            mult = CONDITION_MULTIPLIERS.get(cond_key, Decimal("1.0"))
            qty = r.quantity or 1
            adj_value = (
                Decimal(str(r.market_price)) * mult * qty
                if r.market_price
                else ""
            )
            writer.writerow([
                r.id, r.card_id, r.card_name, r.set_id, r.rarity,
                r.quantity, r.condition, r.grade_authority, r.grade,
                r.acquisition_price, r.acquisition_date, r.notes,
                r.market_price, r.latest_price_date,
                float(mult), adj_value,
            ])

        csv_text = output.getvalue()

        outfile = args.output or "inventory_export.csv"
        with open(outfile, "w", newline="") as f:
            f.write(csv_text)
        print(f"Exported {len(rows)} items to {outfile}")


# ---------------------------------------------------------------------------
# valuation -- snapshot collection value to inventory_valuations
# ---------------------------------------------------------------------------


def cmd_valuation(args):
    """Snapshot the current collection value into inventory_valuations."""
    with SessionLocal() as session:
        rows = session.execute(
            text("""
                SELECT
                    ui.id,
                    ui.card_id,
                    dc.name AS card_name,
                    ui.quantity,
                    ui.condition,
                    ui.grade_authority,
                    ui.grade,
                    lp.low_price,
                    lp.mid_price,
                    lp.high_price,
                    lp.market_price
                FROM user_inventory ui
                JOIN dim_cards dc ON dc.card_id = ui.card_id
                LEFT JOIN LATERAL (
                    SELECT low_price, mid_price, high_price, market_price
                    FROM fact_market_prices fmp
                    WHERE fmp.card_id = ui.card_id
                    ORDER BY fmp.price_date DESC
                    LIMIT 1
                ) lp ON true
            """)
        ).fetchall()

        if not rows:
            print("Inventory is empty. Nothing to valuate.")
            return

        total_low = Decimal("0")
        total_mid = Decimal("0")
        total_high = Decimal("0")
        total_cards = 0
        breakdown = []

        for r in rows:
            cond_key = _effective_condition(r)
            mult = CONDITION_MULTIPLIERS.get(cond_key, Decimal("1.0"))
            qty = r.quantity or 1
            total_cards += qty

            entry = {
                "card_id": r.card_id,
                "card_name": r.card_name,
                "quantity": qty,
                "condition": cond_key,
                "multiplier": float(mult),
            }

            if r.low_price:
                val = Decimal(str(r.low_price)) * mult * qty
                total_low += val
                entry["low_value"] = float(val)
            if r.mid_price:
                val = Decimal(str(r.mid_price)) * mult * qty
                total_mid += val
                entry["mid_value"] = float(val)
            elif r.market_price:
                # Fall back to market_price if mid_price is missing
                val = Decimal(str(r.market_price)) * mult * qty
                total_mid += val
                entry["mid_value"] = float(val)
            if r.high_price:
                val = Decimal(str(r.high_price)) * mult * qty
                total_high += val
                entry["high_value"] = float(val)

            breakdown.append(entry)

        today = date.today()
        session.execute(
            text("""
                INSERT INTO inventory_valuations
                    (valuation_date, total_cards, total_value_low,
                     total_value_mid, total_value_high, breakdown)
                VALUES (:vdate, :cards, :low, :mid, :high, :bd)
            """),
            {
                "vdate": today,
                "cards": total_cards,
                "low": total_low,
                "mid": total_mid,
                "high": total_high,
                "bd": json.dumps(breakdown),
            },
        )
        session.commit()

        print(f"=== Valuation Snapshot ({today}) ===")
        print(f"  Total cards: {total_cards:,}")
        print(f"  Value (low):  ${total_low:>12,.2f}")
        print(f"  Value (mid):  ${total_mid:>12,.2f}")
        print(f"  Value (high): ${total_high:>12,.2f}")
        print("\nSnapshot saved to inventory_valuations.")


# ---------------------------------------------------------------------------
# watch -- watch folder for new card images and auto-scan them
# ---------------------------------------------------------------------------


def cmd_watch(args):
    """Watch folder for new card images and auto-scan them."""
    from cardprice.scrapers.watch_folder import watch
    watch(watch_dir=args.dir, auto_accept=args.threshold, once=args.once)


def cmd_server(args):
    """Start the card scanner HTTP server."""
    from cardprice.server import run_server
    run_server(host=args.host, port=args.port)


# ---------------------------------------------------------------------------
# build-*-index -- build ML reference indexes from downloaded card images
# ---------------------------------------------------------------------------


def cmd_build_hash_index(args):
    """Build perceptual hash database from card images."""
    from cardprice.ml.hash_matcher import build_hash_database

    print(f"Building hash database from {args.image_dir}...")
    result = build_hash_database(args.image_dir, args.output)
    print(f"Done: {len(result)} cards indexed")


def cmd_build_dino_index(args):
    """Build DINOv2 FAISS index from card images."""
    from cardprice.ml.dino_matcher import build_reference_index

    print(f"Building DINOv2 FAISS index from {args.image_dir}...")
    count = build_reference_index(args.image_dir, args.index_path, args.mapping_path)
    print(f"Done: {count} cards indexed")


def cmd_build_clip_index(args):
    """Build CLIP embedding indexes (text, image, or both)."""
    if args.mode in ("text", "both"):
        from cardprice.ml.clip_matcher import build_text_index

        with SessionLocal() as session:
            print("Building CLIP text index...")
            path = build_text_index(session)
            print(f"Text index saved to {path}")
    if args.mode in ("image", "both"):
        from cardprice.ml.clip_matcher import build_image_index

        print(f"Building CLIP image index from {args.image_dir}...")
        path = build_image_index(args.image_dir)
        print(f"Image index saved to {path}")


def cmd_telegram(args):
    """Start Telegram card scanner bot."""
    from cardprice.telegram_bot import main as tg_main
    tg_main()


# ---------------------------------------------------------------------------
# main -- argparse setup and dispatch
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        prog="cardprice",
        description="Cardprice data pipeline CLI",
    )
    sub = parser.add_subparsers(dest="command")

    # refresh
    sub.add_parser("refresh", help="Run weekly refresh (live prices + card catalog)")

    # backfill
    bf = sub.add_parser("backfill", help="Run historical price backfill")
    bf.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    bf.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")

    # migrate
    sub.add_parser("migrate", help="Run DB migrations (create tables)")

    # mapping
    sub.add_parser("mapping", help="Run TCGCSV product-to-card mapping")

    # pokeapi
    sub.add_parser("pokeapi", help="Fetch Pokemon species metadata from PokeAPI")

    # download-images
    dl = sub.add_parser("download-images", help="Download card images")
    dl.add_argument(
        "--size",
        choices=["small", "large"],
        default="small",
        help="Image size to download (default: small)",
    )
    dl.add_argument(
        "--output",
        default="data/card_images",
        help="Output directory (default: data/card_images)",
    )
    dl.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="DB fetch batch size (default: 50)",
    )

    # priority
    pr = sub.add_parser("priority", help="Build card scrape priority queue")
    pr.add_argument(
        "--top-n",
        type=int,
        default=2000,
        help="Number of top cards to score (default: 2000)",
    )

    # smogon
    sm = sub.add_parser("smogon", help="Fetch Smogon competitive usage stats")
    sm.add_argument(
        "--formats",
        type=str,
        default=None,
        help="Comma-separated format slugs (default: gen9ou,gen9uu,gen9ubers,gen9vgc2025)",
    )

    # status
    sub.add_parser("status", help="Show DB row counts and date ranges")

    # scan
    sc = sub.add_parser("scan", help="Scan card image to identify")
    sc.add_argument("image_path", nargs="?", default=None, help="Path to card image")
    sc.add_argument("--dir", type=str, default=None, help="Directory of card images")
    sc.add_argument(
        "--model",
        choices=["claude-haiku-4-5", "clip", "dino", "hash", "cascade"],
        default="claude-haiku-4-5",
        help="Identification model (default: claude-haiku-4-5)",
    )
    sc.add_argument("--add", action="store_true", help="Auto-add identified cards to inventory")
    sc.add_argument("--threshold", type=float, default=0.85, help="Confidence threshold for auto-add (default: 0.85)")

    # inventory (with sub-subcommands)
    inv = sub.add_parser("inventory", help="Manage card inventory")
    inv_sub = inv.add_subparsers(dest="inv_action")

    inv_sub.add_parser("list", help="Show all cards with current values")

    inv_add = inv_sub.add_parser("add", help="Manually add a card")
    inv_add.add_argument("card_id", help="Card ID (e.g. base1-4/holofoil)")
    inv_add.add_argument(
        "--condition", default="NM", help="Condition (NM/LP/MP/HP/DMG)"
    )
    inv_add.add_argument(
        "--price", type=float, default=None, help="Acquisition price"
    )
    inv_add.add_argument(
        "--quantity", type=int, default=1, help="Quantity (default: 1)"
    )
    inv_add.add_argument("--notes", type=str, default=None, help="Notes")

    inv_rm = inv_sub.add_parser("remove", help="Remove a card by inventory ID")
    inv_rm.add_argument("id", type=int, help="Inventory row ID to remove")

    inv_sub.add_parser(
        "value", help="Total collection value with condition multipliers"
    )

    inv_exp = inv_sub.add_parser("export", help="Export inventory to file")
    inv_exp.add_argument(
        "--format", default="csv", help="Export format (default: csv)"
    )
    inv_exp.add_argument(
        "--output", type=str, default=None, help="Output file path"
    )

    # watch
    wa = sub.add_parser("watch", help="Watch folder for card images to auto-scan")
    wa.add_argument("--dir", default="data/inbox", help="Directory to watch (default: data/inbox)")
    wa.add_argument("--threshold", type=float, default=0.85, help="Auto-accept confidence threshold (default: 0.85)")
    wa.add_argument("--once", action="store_true", help="Process existing images and exit (don't loop)")

    # valuation
    sub.add_parser("valuation", help="Snapshot collection value to DB")

    # server
    sv = sub.add_parser("server", help="Start card scanner HTTP server")
    sv.add_argument("--port", type=int, default=8888, help="Port (default: 8888)")
    sv.add_argument("--host", default="0.0.0.0", help="Host (default: 0.0.0.0)")

    # build-hash-index
    bhi = sub.add_parser("build-hash-index", help="Build perceptual hash database from card images")
    bhi.add_argument("--image-dir", default="data/card_images", help="Card images directory")
    bhi.add_argument("--output", default="data/hash_db.pkl", help="Output hash DB path")

    # build-dino-index
    bdi = sub.add_parser("build-dino-index", help="Build DINOv2 FAISS index from card images")
    bdi.add_argument("--image-dir", default="data/card_images", help="Card images directory")
    bdi.add_argument("--index-path", default="data/dino_index.faiss", help="Output FAISS index path")
    bdi.add_argument("--mapping-path", default="data/dino_card_ids.pkl", help="Output card ID mapping path")

    # build-clip-index
    bci = sub.add_parser("build-clip-index", help="Build CLIP embedding indexes")
    bci.add_argument("--mode", choices=["text", "image", "both"], default="text", help="Index type to build")
    bci.add_argument("--image-dir", default="data/card_images", help="Card images directory (for image mode)")

    # telegram
    sub.add_parser("telegram", help="Start Telegram card scanner bot")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    dispatch = {
        "refresh": cmd_refresh,
        "backfill": cmd_backfill,
        "migrate": cmd_migrate,
        "mapping": cmd_mapping,
        "pokeapi": cmd_pokeapi,
        "download-images": cmd_download_images,
        "priority": cmd_priority,
        "smogon": cmd_smogon,
        "status": cmd_status,
        "scan": cmd_scan,
        "inventory": cmd_inventory,
        "valuation": cmd_valuation,
        "watch": cmd_watch,
        "server": cmd_server,
        "build-hash-index": cmd_build_hash_index,
        "build-dino-index": cmd_build_dino_index,
        "build-clip-index": cmd_build_clip_index,
        "telegram": cmd_telegram,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
