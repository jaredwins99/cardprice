#!/bin/bash
#
# auto_resolve.sh: Watch data/pending_scans/ for new .json files and attempt
# to auto-resolve them using the ML cascade with relaxed thresholds.
#
# Normal cascade thresholds: DINOv2 > 0.65, CLIP > 0.75
# Relaxed thresholds here:   DINOv2 > 0.45, CLIP > 0.55
#
# This catches cards the server rejected at normal thresholds but that are
# still likely correct with the relaxed ones.
#
# Usage:
#   ./scripts/auto_resolve.sh          # watch mode (runs forever)
#   ./scripts/auto_resolve.sh --once   # process existing pending files and exit
#
# Requires: inotifywait (sudo apt install inotify-tools)
#

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PENDING_DIR="$PROJECT_DIR/data/pending_scans"
LOG_PREFIX="[auto_resolve]"

mkdir -p "$PENDING_DIR"

log() {
    echo "$LOG_PREFIX [$(date +'%Y-%m-%d %H:%M:%S')] $*"
}

# Attempt to resolve a single pending scan JSON file using relaxed thresholds.
try_resolve() {
    local json_file="$1"

    if [ ! -f "$json_file" ]; then
        return
    fi

    python3 - "$json_file" "$PROJECT_DIR" <<'PYEOF'
import json
import sys
import os

json_file = sys.argv[1]
project_dir = sys.argv[2]

# Add project root to path so imports work
sys.path.insert(0, project_dir)

try:
    data = json.loads(open(json_file).read())
except (json.JSONDecodeError, OSError) as e:
    print(f"  ERROR: could not read {json_file}: {e}", file=sys.stderr)
    sys.exit(1)

# Skip if already resolved or missing image
if data.get("status") != "pending":
    print(f"  SKIP: {json_file} is not pending (status={data.get('status')})")
    sys.exit(0)

image_path = data.get("image_path")
if not image_path or not os.path.isfile(image_path):
    print(f"  SKIP: image not found at {image_path}")
    sys.exit(0)

scan_id = data.get("scan_id", os.path.basename(json_file).replace(".json", ""))
print(f"  Processing {scan_id} -> {image_path}")

# --- Relaxed-threshold cascade (skip hash tier, it already ran in the server) ---

DINO_THRESHOLD = 0.45
CLIP_THRESHOLD = 0.55

from pathlib import Path
_PROJECT_ROOT = Path(project_dir)
_DINO_INDEX_PATH = _PROJECT_ROOT / "data" / "dino_index.faiss"
_CLIP_IMAGE_INDEX_PATH = _PROJECT_ROOT / "data" / "clip_image_index.pkl"

card_id = None
confidence = 0.0
method = None

# Tier 2: DINOv2 + FAISS (relaxed)
try:
    if _DINO_INDEX_PATH.exists():
        from cardprice.ml.dino_matcher import identify_card as dino_identify
        matches = dino_identify(image_path)
        if matches:
            raw_cid = matches[0][0]
            parts = raw_cid.split("/", 1)
            cid = parts[1] if len(parts) > 1 else raw_cid
            sim = float(matches[0][1])
            if sim > DINO_THRESHOLD:
                card_id = cid
                confidence = sim
                method = "dino_relaxed"
                print(f"  DINOv2 match: {card_id} ({sim:.2%})")
            else:
                print(f"  DINOv2 best: {cid} ({sim:.2%}) < threshold {DINO_THRESHOLD:.0%}")
except Exception as e:
    print(f"  DINOv2 error: {e}", file=sys.stderr)

# Tier 2.5: CLIP image-to-image (relaxed), only if DINOv2 did not match
if card_id is None:
    try:
        if _CLIP_IMAGE_INDEX_PATH.exists():
            from cardprice.ml.clip_matcher import identify_card_by_image
            matches = identify_card_by_image(image_path, index_path=str(_CLIP_IMAGE_INDEX_PATH))
            if matches:
                raw_cid = matches[0][0]
                parts = raw_cid.split("/", 1)
                cid = parts[1] if len(parts) > 1 and "/" in parts[1] else raw_cid
                sim = float(matches[0][1])
                if sim > CLIP_THRESHOLD:
                    card_id = cid
                    confidence = sim
                    method = "clip_relaxed"
                    print(f"  CLIP match: {card_id} ({sim:.2%})")
                else:
                    print(f"  CLIP best: {cid} ({sim:.2%}) < threshold {CLIP_THRESHOLD:.0%}")
    except Exception as e:
        print(f"  CLIP error: {e}", file=sys.stderr)

# --- Update the JSON if we got a match ---
if card_id is not None:
    data["status"] = "resolved"
    data["card_id"] = card_id
    data["confidence"] = confidence
    data["method"] = method
    with open(json_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  RESOLVED: {scan_id} -> {card_id} ({confidence:.0%}) via {method}")
else:
    print(f"  NO MATCH: {scan_id} still pending")
PYEOF
}

# --once mode: process all existing pending files and exit
if [ "${1:-}" = "--once" ]; then
    log "Running in --once mode: processing existing pending files..."
    count=0
    for f in "$PENDING_DIR"/*.json; do
        [ -f "$f" ] || continue
        log "Checking $(basename "$f")"
        try_resolve "$f"
        count=$((count + 1))
    done
    log "Done. Checked $count file(s)."
    exit 0
fi

# Watch mode: use inotifywait
if ! command -v inotifywait &>/dev/null; then
    echo "ERROR: inotifywait not found. Install with: sudo apt install inotify-tools" >&2
    exit 1
fi

log "Watching $PENDING_DIR for new .json files..."
log "Relaxed thresholds: DINOv2 > 0.45, CLIP > 0.55"
log "Press Ctrl+C to stop."

# First pass: process any existing pending files
for f in "$PENDING_DIR"/*.json; do
    [ -f "$f" ] || continue
    log "Checking existing: $(basename "$f")"
    try_resolve "$f"
done

# Watch loop
inotifywait -m -e create -e moved_to --format '%f' "$PENDING_DIR" |
while read -r filename; do
    if [[ "$filename" == *.json ]]; then
        log "New file detected: $filename"
        # Brief pause for the server to finish writing the file
        sleep 0.5
        try_resolve "$PENDING_DIR/$filename"
    fi
done
