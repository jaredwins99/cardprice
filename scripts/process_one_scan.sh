#!/bin/bash
#
# process_one_scan.sh: Helper for Claude Code agent to read a pending scan.
#
# Takes a scan_id as argument, reads the pending JSON metadata,
# and prints the image path (for the agent to read via Read tool).
#
# Usage:
#   ./scripts/process_one_scan.sh <scan_id>
#
# Example:
#   ./scripts/process_one_scan.sh 20260228_174530
#
# Output: JSON with scan metadata and image path
#

if [ -z "$1" ]; then
    echo "Usage: $0 <scan_id>"
    echo "Example: $0 20260228_174530"
    exit 1
fi

SCAN_ID="$1"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PENDING_FILE="$PROJECT_DIR/data/pending_scans/${SCAN_ID}.json"

if [ ! -f "$PENDING_FILE" ]; then
    echo "ERROR: Scan metadata not found at $PENDING_FILE" >&2
    exit 1
fi

# Output the metadata so the agent can read it
cat "$PENDING_FILE"
