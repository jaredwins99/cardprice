"""Card scanner using Claude's vision API to identify Pokemon cards from images."""

import base64
import json
import logging
import mimetypes
import os
import time
from pathlib import Path

import anthropic
from sqlalchemy import text

from cardprice.db.session import SessionLocal

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

SCAN_PROMPT = """\
You are a Pokemon trading card identification expert. Analyze this card image \
and extract all identifiable information. Return ONLY valid JSON with these fields:

{
  "card_name": "full card name as printed",
  "pokemon_name": "Pokemon name only (null if not a Pokemon card, e.g. Trainer/Energy)",
  "set_name": "set name from the card's set symbol or text",
  "card_number": "collector number as printed (e.g. '4/102', '025/185')",
  "rarity": "one of: Common, Uncommon, Rare, Rare Holo, Rare Holo EX, Rare Ultra, \
Rare Secret, Rare Rainbow, Illustration Rare, Special Illustration Rare, \
Hyper Rare, Promo, or the exact rarity if identifiable",
  "edition": "1st Edition, Unlimited, Shadowless, or null",
  "condition": "estimated condition: NM, LP, MP, HP, or DMG based on visible wear",
  "is_holographic": true or false,
  "is_graded": true or false,
  "grading_authority": "PSA, BGS, CGC, or null",
  "grade": "numeric grade as string (e.g. '10', '9.5') or null",
  "language": "English, Japanese, etc.",
  "confidence": 0.0 to 1.0 overall confidence in identification,
  "notes": "any additional observations about the card"
}

Be precise with card_number - include both the card number and set total if visible \
(e.g. "4/102"). If a field cannot be determined, use null. For confidence, consider \
how clearly you can read the card details.\
"""


def _load_image_b64(image_path: str | Path) -> tuple[str, str]:
    """Load an image file and return (base64_data, media_type)."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image format: {path.suffix}")

    media_type, _ = mimetypes.guess_type(str(path))
    if media_type is None:
        media_type = "image/jpeg"

    data = path.read_bytes()
    return base64.standard_b64encode(data).decode("ascii"), media_type


def _parse_json_response(text_content: str) -> dict:
    """Extract JSON from Claude's response, handling markdown code fences."""
    stripped = text_content.strip()
    # Strip ```json ... ``` wrappers if present
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        # Drop first line (```json) and last line (```)
        lines = [l for l in lines[1:] if l.strip() != "```"]
        stripped = "\n".join(lines)
    return json.loads(stripped)


def scan_card(image_path: str | Path, model: str = "claude-haiku-4-5") -> dict:
    """Scan a single card image using Claude vision and return structured data.

    Args:
        image_path: Path to the card image file.
        model: Anthropic model to use (default claude-haiku-4-5 for cost efficiency).

    Returns:
        dict with fields: card_name, pokemon_name, set_name, card_number, rarity,
        edition, condition, is_holographic, is_graded, grading_authority, grade,
        language, confidence, notes.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set")

    image_b64, media_type = _load_image_b64(image_path)

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": SCAN_PROMPT,
                    },
                ],
            }
        ],
    )

    response_text = message.content[0].text
    result = _parse_json_response(response_text)

    # Attach metadata
    result["_image_path"] = str(image_path)
    result["_model"] = model
    result["_raw_response"] = response_text

    return result


def scan_batch(
    image_dir: str | Path, model: str = "claude-haiku-4-5"
) -> list[dict]:
    """Scan all card images in a directory.

    Rate-limited to ~1-2 requests per second to stay within API limits.

    Args:
        image_dir: Directory containing card images.
        model: Anthropic model to use.

    Returns:
        List of scan result dicts (one per image). Failed scans include an
        '_error' key instead of card data.
    """
    directory = Path(image_dir)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    image_files = sorted(
        f for f in directory.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_files:
        logger.warning("No image files found in %s", directory)
        return []

    logger.info("Scanning %d images from %s", len(image_files), directory)
    results = []

    for i, img_path in enumerate(image_files):
        logger.info("[%d/%d] Scanning %s", i + 1, len(image_files), img_path.name)
        try:
            result = scan_card(img_path, model=model)
            results.append(result)
        except Exception as e:
            logger.error("Failed to scan %s: %s", img_path.name, e)
            results.append({"_image_path": str(img_path), "_error": str(e)})

        # Rate limit: sleep between requests (skip after last image)
        if i < len(image_files) - 1:
            time.sleep(0.7)

    logger.info("Batch complete: %d scanned, %d errors",
                sum(1 for r in results if "_error" not in r),
                sum(1 for r in results if "_error" in r))
    return results


def match_to_database(scan_result: dict, session=None) -> tuple[str | None, float]:
    """Match a scan result against dim_cards to find the best card_id.

    Uses card name, set, and card number for matching. Falls back to
    progressively looser matches if exact match fails.

    Args:
        scan_result: Dict returned by scan_card().
        session: SQLAlchemy session. If None, creates one from SessionLocal.

    Returns:
        (card_id, confidence) tuple. card_id is None if no match found.
    """
    own_session = session is None
    if own_session:
        session = SessionLocal()

    try:
        card_name = scan_result.get("card_name")
        set_name = scan_result.get("set_name")
        card_number = scan_result.get("card_number")

        if not card_name:
            return None, 0.0

        # Normalize card number: "4/102" -> "4"
        number_only = None
        if card_number:
            number_only = card_number.split("/")[0].strip().lstrip("0") or "0"

        # Strategy 1: exact match on name + set + number
        if set_name and number_only:
            row = session.execute(
                text("""
                    SELECT c.card_id
                    FROM dim_cards c
                    JOIN dim_sets s ON c.set_id = s.set_id
                    WHERE LOWER(c.name) = LOWER(:name)
                      AND LOWER(s.name) = LOWER(:set_name)
                      AND LTRIM(c.card_number, '0') = :number
                    LIMIT 1
                """),
                {"name": card_name, "set_name": set_name, "number": number_only},
            ).fetchone()
            if row:
                return row[0], 0.95

        # Strategy 2: name + number (set name might be slightly off)
        if number_only:
            row = session.execute(
                text("""
                    SELECT c.card_id
                    FROM dim_cards c
                    WHERE LOWER(c.name) = LOWER(:name)
                      AND LTRIM(c.card_number, '0') = :number
                    LIMIT 1
                """),
                {"name": card_name, "number": number_only},
            ).fetchone()
            if row:
                return row[0], 0.80

        # Strategy 3: name + set (number might be misread)
        if set_name:
            row = session.execute(
                text("""
                    SELECT c.card_id
                    FROM dim_cards c
                    JOIN dim_sets s ON c.set_id = s.set_id
                    WHERE LOWER(c.name) = LOWER(:name)
                      AND LOWER(s.name) = LOWER(:set_name)
                    LIMIT 1
                """),
                {"name": card_name, "set_name": set_name},
            ).fetchone()
            if row:
                return row[0], 0.70

        # Strategy 4: name only with fuzzy-ish match via LIKE
        row = session.execute(
            text("""
                SELECT c.card_id
                FROM dim_cards c
                WHERE LOWER(c.name) = LOWER(:name)
                LIMIT 1
            """),
            {"name": card_name},
        ).fetchone()
        if row:
            return row[0], 0.50

        return None, 0.0

    finally:
        if own_session:
            session.close()


def add_to_inventory(
    scan_result: dict,
    card_id: str,
    session=None,
    *,
    quantity: int = 1,
    acquisition_source: str | None = None,
    acquisition_price: float | None = None,
) -> tuple[int, int]:
    """Insert a scanned card into user_inventory and inventory_scans.

    Args:
        scan_result: Dict returned by scan_card().
        card_id: Matched card_id from match_to_database().
        session: SQLAlchemy session. If None, creates one from SessionLocal.
        quantity: Number of copies (default 1).
        acquisition_source: One of 'pulled', 'purchased', 'traded', or None.
        acquisition_price: Price paid, if known.

    Returns:
        (inventory_id, scan_id) tuple of the inserted row IDs.
    """
    own_session = session is None
    if own_session:
        session = SessionLocal()

    try:
        # Map condition from scan to the CHECK constraint values
        condition = scan_result.get("condition")
        valid_conditions = {"NM", "LP", "MP", "HP", "DMG"}
        if condition not in valid_conditions:
            condition = None

        # Map grading authority
        grade_authority = scan_result.get("grading_authority")
        valid_authorities = {"PSA", "BGS", "CGC"}
        if grade_authority not in valid_authorities:
            grade_authority = None

        grade = scan_result.get("grade") if grade_authority else None
        confidence = scan_result.get("confidence", 0.0)
        image_path = scan_result.get("_image_path")

        # Insert into inventory_scans
        scan_row = session.execute(
            text("""
                INSERT INTO inventory_scans
                    (image_path, identified_card_id, identified_condition,
                     confidence, model_used, raw_response, accepted)
                VALUES
                    (:image_path, :card_id, :condition,
                     :confidence, :model, :raw_response, TRUE)
                RETURNING id
            """),
            {
                "image_path": image_path,
                "card_id": card_id,
                "condition": condition,
                "confidence": confidence,
                "model": scan_result.get("_model"),
                "raw_response": json.dumps(
                    {k: v for k, v in scan_result.items() if not k.startswith("_")}
                ),
            },
        ).fetchone()
        scan_id = scan_row[0]

        # Insert into user_inventory
        inv_row = session.execute(
            text("""
                INSERT INTO user_inventory
                    (card_id, quantity, condition, grade_authority, grade,
                     acquisition_price, acquisition_source, image_path,
                     scan_confidence, notes)
                VALUES
                    (:card_id, :quantity, :condition, :grade_authority, :grade,
                     :acquisition_price, :acquisition_source, :image_path,
                     :scan_confidence, :notes)
                RETURNING id
            """),
            {
                "card_id": card_id,
                "quantity": quantity,
                "condition": condition,
                "grade_authority": grade_authority,
                "grade": grade,
                "acquisition_price": acquisition_price,
                "acquisition_source": acquisition_source,
                "image_path": image_path,
                "scan_confidence": confidence,
                "notes": scan_result.get("notes"),
            },
        ).fetchone()
        inventory_id = inv_row[0]

        session.commit()
        logger.info("Added card %s to inventory (inv=%d, scan=%d)",
                     card_id, inventory_id, scan_id)
        return inventory_id, scan_id

    except Exception:
        session.rollback()
        raise
    finally:
        if own_session:
            session.close()
