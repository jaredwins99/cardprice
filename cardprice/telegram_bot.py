"""Telegram bot for Pokemon card scanning.

Setup:
1. Create a bot via @BotFather on Telegram
2. Set TELEGRAM_BOT_TOKEN env var
3. Run: python -m cardprice.telegram_bot

Then send card photos to your bot — it'll identify them and show prices.
"""

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

INBOX_DIR = Path("data/inbox")
AUTO_ACCEPT = 0.85


async def handle_photo(update, context):
    """Handle incoming photo messages."""
    photo = update.message.photo[-1]  # highest resolution
    file = await context.bot.get_file(photo.file_id)

    # Save to inbox
    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    save_path = INBOX_DIR / f"tg_{update.message.message_id}.jpg"
    await file.download_to_drive(str(save_path))

    await update.message.reply_text("Scanning...")

    try:
        from cardprice.ml import identify_card
        from cardprice.db.session import SessionLocal
        from sqlalchemy import text

        with SessionLocal() as session:
            result = identify_card(str(save_path), session=session)

            if result["card_id"]:
                # Look up card details
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
                method = result["method"] or "?"

                msg = f"*{name}*\nSet: {set_name}\nPrice: {price}\nConfidence: {conf} ({method})"

                # Auto-add to inventory
                if result["confidence"] >= AUTO_ACCEPT:
                    session.execute(
                        text("""
                            INSERT INTO user_inventory (card_id, quantity, condition, notes)
                            VALUES (:cid, 1, 'NM', :notes)
                        """),
                        {"cid": result["card_id"], "notes": f"Telegram scan {conf}"},
                    )
                    session.commit()
                    msg += "\n\nAdded to inventory!"

                await update.message.reply_text(msg, parse_mode="Markdown")
            else:
                await update.message.reply_text("Could not identify this card. Try a clearer photo.")

    except Exception as e:
        logger.error("Scan error: %s", e)
        await update.message.reply_text(f"Error: {e}")


async def handle_start(update, context):
    """Handle /start command."""
    await update.message.reply_text(
        "Send me a photo of a Pokemon card and I'll identify it and tell you the price!"
    )


def main():
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

    print("Telegram bot running. Send card photos to scan!")
    app.run_polling()


if __name__ == "__main__":
    main()
