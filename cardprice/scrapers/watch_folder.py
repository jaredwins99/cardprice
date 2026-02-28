"""Watch a folder for new card images and auto-scan them.

Usage:
    python -m cardprice.cli watch --dir ~/cardprice-inbox

When a new image appears in the watched directory:
1. Run the cascade identification pipeline
2. If confidence >= threshold, auto-add to inventory
3. Move processed images to done/ subfolder
4. Log results
"""

import json
import logging
import shutil
import time
from pathlib import Path

from sqlalchemy import text

from cardprice.db.session import SessionLocal
from cardprice.utils.image_convert import ensure_compatible

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic"}
DEFAULT_WATCH_DIR = "data/inbox"
DEFAULT_DONE_DIR = "data/inbox/done"
DEFAULT_FAILED_DIR = "data/inbox/failed"
POLL_INTERVAL = 2.0  # seconds
AUTO_ACCEPT_THRESHOLD = 0.85


def _find_new_images(watch_dir: Path, done_dir: Path, failed_dir: Path) -> list[Path]:
    """Find image files in watch_dir that haven't been processed."""
    images = []
    for f in sorted(watch_dir.iterdir()):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(f)
    return images


def _process_image(image_path: Path, session, auto_accept: float) -> dict:
    """Scan a single image and optionally add to inventory."""
    from cardprice.ml import identify_card

    compatible_path = ensure_compatible(str(image_path))
    result = identify_card(compatible_path, session=session)

    output = {
        "image": image_path.name,
        "card_id": result["card_id"],
        "confidence": result["confidence"],
        "method": result["method"],
        "added_to_inventory": False,
    }

    if result["card_id"] and result["confidence"] >= auto_accept:
        # Auto-add to inventory
        try:
            session.execute(
                text("""
                    INSERT INTO user_inventory (card_id, quantity, condition, notes)
                    VALUES (:cid, 1, 'NM', :notes)
                """),
                {
                    "cid": result["card_id"],
                    "notes": f"Auto-scanned from {image_path.name} via {result['method']} (conf={result['confidence']:.2f})",
                },
            )
            # Record scan
            session.execute(
                text("""
                    INSERT INTO inventory_scans
                        (image_path, identified_card_id, confidence, model_used, raw_response, accepted)
                    VALUES (:path, :cid, :conf, :model, :raw, TRUE)
                """),
                {
                    "path": str(image_path),
                    "cid": result["card_id"],
                    "conf": result["confidence"],
                    "model": result["method"],
                    "raw": json.dumps({"method": result["method"]}),
                },
            )
            session.commit()
            output["added_to_inventory"] = True

            # Look up card name for display
            row = session.execute(
                text("SELECT name, set_id FROM dim_cards WHERE card_id = :cid"),
                {"cid": result["card_id"]},
            ).fetchone()
            if row:
                output["card_name"] = row.name
                output["set_id"] = row.set_id

        except Exception as e:
            session.rollback()
            logger.error("Failed to add to inventory: %s", e)

    return output


def watch(
    watch_dir: str = DEFAULT_WATCH_DIR,
    auto_accept: float = AUTO_ACCEPT_THRESHOLD,
    once: bool = False,
):
    """Watch a directory for new card images and process them.

    Args:
        watch_dir: Directory to watch for new images.
        auto_accept: Minimum confidence to auto-add to inventory.
        once: If True, process existing images and exit (don't loop).
    """
    watch_path = Path(watch_dir)
    done_path = watch_path / "done"
    failed_path = watch_path / "failed"

    watch_path.mkdir(parents=True, exist_ok=True)
    done_path.mkdir(exist_ok=True)
    failed_path.mkdir(exist_ok=True)

    logger.info("Watching %s for card images (auto-accept >= %.0f%%)", watch_path, auto_accept * 100)

    with SessionLocal() as session:
        while True:
            images = _find_new_images(watch_path, done_path, failed_path)

            for img in images:
                logger.info("Processing: %s", img.name)
                try:
                    result = _process_image(img, session, auto_accept)

                    if result["card_id"]:
                        status = "ADDED" if result["added_to_inventory"] else "MATCHED (below threshold)"
                        name = result.get("card_name", result["card_id"])
                        print(f"  {status}: {name} ({result['confidence']:.0%} via {result['method']})")
                        shutil.move(str(img), str(done_path / img.name))
                    else:
                        print(f"  NO MATCH: {img.name}")
                        shutil.move(str(img), str(failed_path / img.name))

                except Exception as e:
                    logger.error("Error processing %s: %s", img.name, e)
                    shutil.move(str(img), str(failed_path / img.name))

            if once:
                break

            time.sleep(POLL_INTERVAL)
