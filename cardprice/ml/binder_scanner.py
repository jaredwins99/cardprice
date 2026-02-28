"""Multi-card Pokemon card identification from binder page photos.

This module identifies all visible Pokemon trading cards in a single binder
page photo, returning structured JSON with card details and per-card confidence
scores. Optimized for speed and cost efficiency.
"""

import base64
import json
import logging
import mimetypes
import os
from pathlib import Path
from typing import Optional

import anthropic

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

# ============================================================================
# Prompt Templates
# ============================================================================

BINDER_SCAN_PROMPT = """\
You are a Pokemon trading card identification expert analyzing a binder page photo.
Your task is to identify ALL visible Pokemon trading cards in the image.

CRITICAL CONSTRAINTS:
- Return ONLY valid JSON. No explanations, preamble, or notes outside the JSON.
- Include every card you can clearly see (partial cards are OK if you can identify them).
- For each card, extract: name, Pokemon species, set name, card number, rarity.
- Ignore damaged/corrupted binder sleeves, background reflections, and out-of-focus cards.
- If a card is too blurry or obscured, mark it with confidence < 0.5 and set most fields to null.

IDENTIFICATION TIPS:
- Set name: Look for the set symbol (tiny icon) in the bottom-left of each card.
- Card number: Read the number in the bottom-right (e.g., "25/102").
- Rarity: Star icon bottom-right indicates rarity (1 star=Common, 2=Uncommon, 3=Rare, 3.5=Rare Holo, etc.).
- Holofoil: Sparkly surface indicates holofoil variant.
- Name: Card name is printed at the top in large text.

Return a JSON object with a "cards" array, one card object per visible card, \
in left-to-right, top-to-bottom order:

{
  "cards": [
    {
      "position": "slot_1_top_left",
      "card_name": "full card name as printed",
      "pokemon_name": "species name or null if Trainer/Energy/Supporter",
      "set_name": "set name from symbol or text",
      "card_number": "number/total (e.g., '25/102')",
      "rarity": "Common, Uncommon, Rare, Rare Holo, Rare Holo EX, etc., or null",
      "is_holographic": true or false,
      "is_reverse_holographic": true or false,
      "edition": "1st Edition, Unlimited, Shadowless, or null",
      "condition": "NM, LP, MP, HP, DMG, or null (estimate from visible wear)",
      "language": "English, Japanese, German, etc.",
      "confidence": 0.0 to 1.0,
      "notes": "any observations: glare, overlap, partial visibility, wear, etc."
    }
  ],
  "page_summary": {
    "total_cards_identified": 9,
    "confident_cards": 8,
    "cards_needing_review": 1,
    "page_confidence": 0.82
  }
}
\
"""

BINDER_SCAN_HIGH_ACCURACY_PROMPT = """\
You are a professional Pokemon card grader analyzing a binder page for valuation.
Identify each card with maximum precision. For each card, assess condition carefully.

Return ONLY valid JSON with this structure:

{
  "cards": [
    {
      "position": "slot_1_top_left",
      "card_name": "exact name, correcting any OCR errors",
      "pokemon_name": "species or null",
      "set_code": "set abbreviation if visible (e.g., 'Base1', 'SVe', 'GYM1')",
      "set_name": "full set name from symbol or text",
      "card_number": "number/total (e.g., '25/102')",
      "rarity": "rarity indicator (Common, Uncommon, Rare, Rare Holo, etc.)",
      "edition": "1st Edition, Unlimited, Shadowless, or null",
      "is_holographic": boolean,
      "is_reverse_holographic": boolean,
      "condition": "NM, LP, MP, HP, or DMG",
      "condition_details": "whitening, creases, bends, centering issues, etc.",
      "language": "English, Japanese, German, etc.",
      "estimated_psa_grade": "numeric estimate (8.5, 9, etc.) or null",
      "grading_notes": "observations that would matter to PSA/BGS/CGC grader",
      "confidence": 0.0 to 1.0
    }
  ],
  "page_summary": {
    "total_cards": 9,
    "average_condition": "LP",
    "high_value_potential": ["card names worth $50+"],
    "recommend_grading": ["cards with confidence > 0.85"]
  }
}
\
"""

BINDER_SCAN_FAST_PROMPT = """\
Quickly identify all visible cards in this binder page. Return ONLY JSON.

{
  "cards": [
    {
      "position": "top_left",
      "card_name": "name",
      "set_name": "set",
      "card_number": "number",
      "confidence": 0.0 to 1.0
    }
  ],
  "page_confidence": 0.85
}
\
"""


# ============================================================================
# Helper Functions
# ============================================================================


def _load_image_b64(image_path: str | Path) -> tuple[str, str]:
    """Load an image file and return (base64_data, media_type).

    Args:
        image_path: Path to the image file.

    Returns:
        Tuple of (base64_encoded_data, media_type_string).

    Raises:
        FileNotFoundError: If image doesn't exist.
        ValueError: If image format is unsupported.
    """
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
    """Extract JSON from Claude's response, handling markdown code fences.

    Args:
        text_content: Raw text response from Claude.

    Returns:
        Parsed JSON as dict.

    Raises:
        json.JSONDecodeError: If JSON parsing fails.
    """
    stripped = text_content.strip()
    # Strip ```json ... ``` wrappers if present
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        # Drop first line (```json) and last line (```)
        lines = [l for l in lines[1:] if l.strip() != "```"]
        stripped = "\n".join(lines)
    return json.loads(stripped)


# ============================================================================
# Main Scanning Functions
# ============================================================================


def scan_binder_page(
    image_path: str | Path,
    model: str = "claude-haiku-4-5",
    accuracy_mode: str = "balanced",
) -> dict:
    """Scan a binder page photo and identify all visible cards.

    Args:
        image_path: Path to binder page image.
        model: Claude model to use. Default claude-haiku-4-5 for cost efficiency.
        accuracy_mode: One of "fast", "balanced", or "high_accuracy".
                       - "fast": minimal details, ~$0.001 per page
                       - "balanced": standard details, ~$0.0015 per page (default)
                       - "high_accuracy": full grading-level details, ~$0.002 per page

    Returns:
        Dict with structure:
        {
            "cards": [
                {
                    "card_name": str,
                    "pokemon_name": str | None,
                    "set_name": str,
                    "card_number": str,
                    "confidence": float,
                    ... (other fields depending on accuracy_mode)
                }
            ],
            "page_summary": {
                "total_cards_identified": int,
                "confident_cards": int,
                "page_confidence": float,
                "cards_needing_review": int
            },
            "_metadata": {
                "image_path": str,
                "model": str,
                "accuracy_mode": str,
                "input_tokens": int,
                "output_tokens": int
            }
        }

    Raises:
        FileNotFoundError: If image doesn't exist.
        ValueError: If image format unsupported or API key missing.
        anthropic.APIError: If API call fails.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set")

    if accuracy_mode not in ("fast", "balanced", "high_accuracy"):
        raise ValueError(f"Invalid accuracy_mode: {accuracy_mode}")

    # Select prompt based on accuracy mode
    if accuracy_mode == "fast":
        prompt = BINDER_SCAN_FAST_PROMPT
        max_tokens = 1024
    elif accuracy_mode == "high_accuracy":
        prompt = BINDER_SCAN_HIGH_ACCURACY_PROMPT
        max_tokens = 2048
    else:  # balanced
        prompt = BINDER_SCAN_PROMPT
        max_tokens = 2048

    image_b64, media_type = _load_image_b64(image_path)

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
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
                        "text": prompt,
                    },
                ],
            }
        ],
    )

    response_text = message.content[0].text
    result = _parse_json_response(response_text)

    # Attach metadata
    result["_metadata"] = {
        "image_path": str(image_path),
        "model": model,
        "accuracy_mode": accuracy_mode,
        "input_tokens": message.usage.input_tokens,
        "output_tokens": message.usage.output_tokens,
        "raw_response": response_text,
    }

    logger.info(
        "Scanned binder page %s: %d cards identified (confidence: %.2f)",
        image_path,
        result.get("page_summary", {}).get("total_cards_identified", 0),
        result.get("page_summary", {}).get("page_confidence", 0.0),
    )

    return result


def scan_binder_batch(
    image_dir: str | Path,
    model: str = "claude-haiku-4-5",
    accuracy_mode: str = "balanced",
) -> list[dict]:
    """Scan all binder page images in a directory.

    Rate-limited to ~1-2 requests per second to stay within API limits.

    Args:
        image_dir: Directory containing binder page images.
        model: Claude model to use.
        accuracy_mode: One of "fast", "balanced", or "high_accuracy".

    Returns:
        List of scan result dicts (one per binder page image). Failed scans
        include an '_error' key instead of card data.
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

    logger.info("Scanning %d binder pages from %s", len(image_files), directory)
    results = []

    import time

    for i, img_path in enumerate(image_files):
        logger.info("[%d/%d] Scanning %s", i + 1, len(image_files), img_path.name)
        try:
            result = scan_binder_page(img_path, model=model, accuracy_mode=accuracy_mode)
            results.append(result)
        except Exception as e:
            logger.error("Failed to scan %s: %s", img_path.name, e)
            results.append({"_image_path": str(img_path), "_error": str(e)})

        # Rate limit: sleep between requests (skip after last image)
        if i < len(image_files) - 1:
            time.sleep(0.7)

    logger.info(
        "Batch complete: %d scanned, %d errors",
        sum(1 for r in results if "_error" not in r),
        sum(1 for r in results if "_error" in r),
    )
    return results


# ============================================================================
# Database Insertion (Optional Convenience Function)
# ============================================================================


def add_binder_scan_to_inventory(
    scan_result: dict,
    session=None,
    *,
    binder_name: Optional[str] = None,
    acquisition_source: Optional[str] = None,
    acquisition_price: Optional[float] = None,
) -> int:
    """Insert a binder scan into the database and link to dim_cards.

    Args:
        scan_result: Dict returned by scan_binder_page().
        session: SQLAlchemy session. Creates one from SessionLocal if None.
        binder_name: Optional name/identifier for the binder.
        acquisition_source: One of 'pulled', 'purchased', 'traded', or None.
        acquisition_price: Price paid for the binder, if known.

    Returns:
        binder_scan_id (int) of the inserted row.

    Raises:
        ValueError: If scan_result lacks required fields.
        sqlalchemy exceptions: If database insert fails.
    """
    from cardprice.db.session import SessionLocal
    from sqlalchemy import text

    own_session = session is None
    if own_session:
        session = SessionLocal()

    try:
        image_path = scan_result.get("_metadata", {}).get("image_path")
        model_used = scan_result.get("_metadata", {}).get("model")
        page_confidence = (
            scan_result.get("page_summary", {}).get("page_confidence", 0.0)
        )
        total_cards = (
            scan_result.get("page_summary", {}).get("total_cards_identified", 0)
        )
        confident_cards = (
            scan_result.get("page_summary", {}).get("confident_cards", 0)
        )

        # Insert into binder_scans (create table if needed)
        scan_row = session.execute(
            text("""
                INSERT INTO binder_scans
                    (image_path, model_used, page_confidence, total_cards,
                     confident_cards, binder_name, acquisition_source,
                     acquisition_price, raw_response)
                VALUES
                    (:image_path, :model, :page_conf, :total, :confident,
                     :binder_name, :acq_source, :acq_price, :raw_resp)
                RETURNING id
            """),
            {
                "image_path": image_path,
                "model": model_used,
                "page_conf": page_confidence,
                "total": total_cards,
                "confident": confident_cards,
                "binder_name": binder_name,
                "acq_source": acquisition_source,
                "acq_price": acquisition_price,
                "raw_resp": json.dumps(scan_result),
            },
        ).fetchone()
        binder_scan_id = scan_row[0]

        # For each card, try to match to dim_cards and insert into binder_page_identifications
        for card in scan_result.get("cards", []):
            card_name = card.get("card_name")
            set_name = card.get("set_name")
            confidence = card.get("confidence", 0.0)

            # Very basic matching: by name + set
            # In production, use the match_to_database() logic from claude_scanner.py
            if card_name and set_name:
                card_row = session.execute(
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

                if card_row:
                    card_id = card_row[0]
                else:
                    card_id = None
            else:
                card_id = None

            condition = card.get("condition")
            valid_conditions = {"NM", "LP", "MP", "HP", "DMG"}
            if condition not in valid_conditions:
                condition = None

            needs_review = confidence < 0.70

            session.execute(
                text("""
                    INSERT INTO binder_page_identifications
                        (binder_scan_id, position, identified_card_id, confidence,
                         condition, needs_review)
                    VALUES
                        (:scan_id, :position, :card_id, :confidence,
                         :condition, :needs_review)
                """),
                {
                    "scan_id": binder_scan_id,
                    "position": card.get("position"),
                    "card_id": card_id,
                    "confidence": confidence,
                    "condition": condition,
                    "needs_review": needs_review,
                },
            )

        session.commit()
        logger.info(
            "Added binder scan to inventory (scan_id=%d, cards=%d)",
            binder_scan_id,
            total_cards,
        )
        return binder_scan_id

    except Exception:
        session.rollback()
        raise
    finally:
        if own_session:
            session.close()
