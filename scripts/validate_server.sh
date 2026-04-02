#!/usr/bin/env bash
# validate_server.sh — Post-edit validation hook for cardprice server
# Called by Claude Code hooks after edits to server.py, slide_scan_v7.py, or ML files.
#
# Usage:
#   validate_server.sh server   — check endpoints on :8888
#   validate_server.sh ml       — smoke-test ML imports
#   validate_server.sh          — (reads stdin JSON from hook, auto-detects mode)
#
# Exit 0 = all pass, exit 1 = any failure.
# Outputs JSON for Claude Code hook consumption.

set -euo pipefail

PROJ="/home/godli/cardprice"
PORT=8888
BASE="http://127.0.0.1:${PORT}"
FAILURES=()
PASSES=()

# ── Helpers ──────────────────────────────────────────────────────────────

pass() { PASSES+=("PASS: $1"); }
fail() { FAILURES+=("FAIL: $1"); }

check_endpoint() {
    local path="$1"
    local expect_in_body="$2"  # substring that MUST appear in response body
    local label="${3:-GET ${path}}"

    local http_code body
    # -s silent, -o body, -w http_code, --max-time 5
    body=$(curl -s --max-time 5 -o - -w '\n%{http_code}' "${BASE}${path}" 2>/dev/null) || {
        fail "${label}: connection refused / timeout"
        return
    }

    http_code="${body##*$'\n'}"
    body="${body%$'\n'*}"

    if [[ "$http_code" != "200" ]]; then
        fail "${label}: HTTP ${http_code} (expected 200)"
        return
    fi

    if [[ -n "$expect_in_body" ]] && ! echo "$body" | grep -qi "$expect_in_body"; then
        fail "${label}: response missing expected content '${expect_in_body}'"
        return
    fi

    pass "${label} -> 200 OK"
}

# ── Determine mode ───────────────────────────────────────────────────────

MODE="${1:-}"

if [[ -z "$MODE" ]]; then
    # Read hook stdin JSON to determine which file was edited
    INPUT=$(cat)
    FILE_PATH=$(echo "$INPUT" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    # PostToolUse for Edit: tool_input.file_path
    # PostToolUse for Write: tool_input.file_path
    fp = d.get('tool_input', {}).get('file_path', '')
    print(fp)
except:
    print('')
" 2>/dev/null || echo "")

    if echo "$FILE_PATH" | grep -qE '(server\.py|slide_scan_v7\.py|page_scanner\.py)'; then
        MODE="server"
    elif echo "$FILE_PATH" | grep -qE 'cardprice/ml/.*\.py$'; then
        MODE="ml"
    else
        # Not a file we care about — pass through silently
        exit 0
    fi
fi

# ── Server endpoint validation ───────────────────────────────────────────

if [[ "$MODE" == "server" ]]; then
    # First check if server is running
    if ! curl -s --max-time 2 "${BASE}/" >/dev/null 2>&1; then
        # Server not running — warn but don't block
        echo '{"hookSpecificOutput":{"hookEventName":"PostToolUse","additionalContext":"WARNING: Server on :8888 is not running. Cannot validate endpoints. Start with: python -m cardprice.server"}}'
        exit 0
    fi

    # Test key GET endpoints with expected content
    check_endpoint "/" "<!DOCTYPE html\|<html\|upload\|scan" "GET / (home page)"
    check_endpoint "/slide-scan-v7" "<!DOCTYPE html\|<html\|slide\|camera\|video" "GET /slide-scan-v7"
    check_endpoint "/page-scanner" "<!DOCTYPE html\|<html\|scanner\|camera\|capture" "GET /page-scanner"
    check_endpoint "/stats" "total_scans\|scan_count\|method" "GET /stats (JSON)"
    check_endpoint "/inventory" "" "GET /inventory"
    check_endpoint "/history" "" "GET /history"
    check_endpoint "/pending" "" "GET /pending"
fi

# ── ML import smoke test ─────────────────────────────────────────────────

if [[ "$MODE" == "ml" ]]; then
    # Quick import check — catches syntax errors, missing deps, circular imports
    IMPORT_OUTPUT=$(cd "$PROJ" && PYTHONPATH="$PROJ" python3 -c "
import sys
errors = []

modules = [
    'cardprice.ml',
    'cardprice.ml.card_segmenter',
    'cardprice.ml.ocr_matcher',
    'cardprice.ml.ref_matcher',
    'cardprice.ml.attack_ocr',
    'cardprice.ml.page_context',
    'cardprice.ml.dino_matcher',
    'cardprice.ml.hp_detector',
    'cardprice.ml.variant_detector',
]

for mod in modules:
    try:
        __import__(mod)
    except Exception as e:
        errors.append(f'{mod}: {type(e).__name__}: {e}')

if errors:
    print('IMPORT_ERRORS')
    for e in errors:
        print(e)
    sys.exit(1)
else:
    print('ALL_IMPORTS_OK')
    sys.exit(0)
" 2>&1) || true

    if echo "$IMPORT_OUTPUT" | grep -q "IMPORT_ERRORS"; then
        # Extract error lines
        while IFS= read -r line; do
            [[ "$line" == "IMPORT_ERRORS" ]] && continue
            [[ -n "$line" ]] && fail "Import: $line"
        done <<< "$IMPORT_OUTPUT"
    elif echo "$IMPORT_OUTPUT" | grep -q "ALL_IMPORTS_OK"; then
        pass "All ML module imports succeeded"
    else
        fail "ML import check produced unexpected output: ${IMPORT_OUTPUT:0:200}"
    fi
fi

# ── Report results ───────────────────────────────────────────────────────

TOTAL=$(( ${#PASSES[@]} + ${#FAILURES[@]} ))

if [[ ${#FAILURES[@]} -gt 0 ]]; then
    # Build failure report for Claude Code
    REPORT="SERVER VALIDATION FAILED (${#FAILURES[@]}/${TOTAL}):"
    for f in "${FAILURES[@]}"; do
        REPORT="${REPORT}\n  ${f}"
    done
    for p in "${PASSES[@]}"; do
        REPORT="${REPORT}\n  ${p}"
    done

    # Output JSON that Claude Code will inject as context
    python3 -c "
import json, sys
report = '''${REPORT}'''
print(json.dumps({
    'hookSpecificOutput': {
        'hookEventName': 'PostToolUse',
        'additionalContext': report
    }
}))
"
    exit 1
else
    # All passed — brief context
    REPORT="Server validation: ${#PASSES[@]}/${TOTAL} checks passed."
    python3 -c "
import json
print(json.dumps({
    'hookSpecificOutput': {
        'hookEventName': 'PostToolUse',
        'additionalContext': '${REPORT}'
    }
}))
"
    exit 0
fi
