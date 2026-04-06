"""Process pending card scans by writing identification results.

This script is meant to be called after a Claude Code agent has identified
the cards. It reads results from stdin (JSON lines) and updates the pending
scan metadata files.

Usage from Claude Code:
    1. Agent reads each pending image via Read tool
    2. Agent identifies the card visually
    3. Agent calls: python scripts/ops/process_pending.py <scan_id> <card_id> [confidence]
"""

import json
import sys
from pathlib import Path

PENDING_DIR = Path("data/pending_scans")


def resolve_scan(scan_id: str, card_id: str, confidence: float = 0.95):
    """Mark a pending scan as resolved with the identified card_id."""
    meta_path = PENDING_DIR / f"{scan_id}.json"
    if not meta_path.exists():
        print(f"ERROR: {meta_path} not found")
        return False

    data = json.loads(meta_path.read_text())
    data["status"] = "resolved"
    data["card_id"] = card_id
    data["confidence"] = confidence
    data["method"] = "claude_code"
    meta_path.write_text(json.dumps(data, indent=2))
    print(f"Resolved {scan_id} -> {card_id} ({confidence:.0%})")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python process_pending.py <scan_id> <card_id> [confidence]")
        sys.exit(1)

    scan_id = sys.argv[1]
    card_id = sys.argv[2]
    conf = float(sys.argv[3]) if len(sys.argv) > 3 else 0.95
    resolve_scan(scan_id, card_id, conf)
