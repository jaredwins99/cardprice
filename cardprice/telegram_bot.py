"""Telegram bot for Pokemon card scanning.

Setup:
1. Create a bot via @BotFather on Telegram
2. Set TELEGRAM_BOT_TOKEN env var
3. Run: python -m cardprice.telegram_bot

Then send card photos to your bot -- it'll identify them and show prices.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INBOX_DIR = PROJECT_ROOT / "data" / "inbox"
AUTO_ACCEPT = 0.85


def _escape_markdown(text: str) -> str:
    """Escape special characters for Telegram MarkdownV2."""
    special = r"_*[]()~`>#+-=|{}.!"
    return "".join(f"\\{c}" if c in special else c for c in text)


async def handle_photo(update, context) -> None:
    """Handle incoming photo messages."""
    if not update.message:
        return

    photo_list = update.message.photo
    if not photo_list:
        return

    photo = photo_list[-1]  # highest resolution
    tg_file = await context.bot.get_file(photo.file_id)

    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    save_path = INBOX_DIR / f"tg_{update.message.message_id}.jpg"
    await tg_file.download_to_drive(str(save_path))

    await update.message.reply_text("Scanning...")

    try:
        from cardprice.ml import identify_card
        from cardprice.db.session import SessionLocal
        from sqlalchemy import text

        with SessionLocal() as session:
            result = identify_card(str(save_path), session=session)

            if not result["card_id"]:
                await update.message.reply_text(
                    "Could not identify this card. Try a clearer photo."
                )
                return

            row = session.execute(
                text("""
                    SELECT c.name, s.name as set_name, p.market_price
                    FROM dim_cards c
                    JOIN dim_sets s ON s.set_id = c.set_id
                    LEFT JOIN LATERAL (
                        SELECT market_price FROM fact_market_prices
                        WHERE card_id = c.card_id
                        ORDER BY price_date DESC LIMIT 1
                    ) p ON true
                    WHERE c.card_id = :cid
                """),
                {"cid": result["card_id"]},
            ).fetchone()

            name = row.name if row else result["card_id"]
            set_name = row.set_name if row else "?"
            price = f"${row.market_price:.2f}" if row and row.market_price else "No price"
            conf = f"{result['confidence']:.0%}"
            method = result.get("method") or "?"

            lines = [
                f"{name}",
                f"Set: {set_name}",
                f"Price: {price}",
                f"Confidence: {conf} ({method})",
            ]

            if result["confidence"] >= AUTO_ACCEPT:
                session.execute(
                    text("""
                        INSERT INTO user_inventory (card_id, quantity, condition, notes)
                        VALUES (:cid, 1, 'NM', :notes)
                    """),
                    {"cid": result["card_id"], "notes": f"Telegram scan {conf}"},
                )
                session.commit()
                lines.append("\nAdded to inventory!")

            await update.message.reply_text("\n".join(lines))

    except Exception:
        logger.exception("Scan error for message %s", update.message.message_id)
        await update.message.reply_text(
            "Something went wrong while scanning. Please try again."
        )


async def handle_start(update, context) -> None:
    """Handle /start command."""
    if not update.message:
        return
    await update.message.reply_text(
        "Send me a photo of a Pokemon card and I'll identify it and tell you the price!"
    )


def main() -> None:
    """Start the Telegram bot."""
    from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        print("Set TELEGRAM_BOT_TOKEN env var first.")
        print("Get one from @BotFather on Telegram.")
        return

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", handle_start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    logger.info("Telegram bot starting. Send card photos to scan!")
    app.run_polling()


if __name__ == "__main__":
    main()
