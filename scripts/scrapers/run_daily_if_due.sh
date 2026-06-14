#!/usr/bin/env bash
# run_daily_if_due.sh — fire a daily scraper job at most once per UTC day.
#
# Usage:
#   ./scripts/scrapers/run_daily_if_due.sh <job> [extra args...]
#   <job> ∈ { tcgplayer, justtcg, tcgcsv_jp, all }
#
# Mechanics:
#   • Per-job UTC-dated lockfile in data/.daily_<job>_done_<YYYY-MM-DD>
#     marks "today's run already kicked off (and completed) successfully."
#     If present, this script is a fast no-op.
#   • Per-job flock side-lock in data/.daily_<job>.flock prevents the cron
#     entry and the systemd-user boot timer from racing into a double-run
#     during the ~5-hour scrape window. flock is non-blocking (-n): the
#     second caller exits 0 immediately rather than queueing.
#   • UTC is used so the lockfile rolls at the same moment as cron's
#     `0 11 * * *` clock (cron here runs in the system TZ but the
#     scrape_log table records UTC; aligning on UTC keeps both consistent
#     and avoids DST edge cases).
#
# This wrapper does NOT modify the scraper scripts themselves. The
# scrape_log table inside the underlying scripts is still authoritative
# for "did product X get scraped" — the lockfile is only authoritative
# for "did we KICK OFF today's daily job yet."

set -u

REPO_ROOT="/home/godli/cardprice"
PY="/home/godli/miniconda3/bin/python"
LOG_DIR="$REPO_ROOT/data/logs"
LOCK_DIR="$REPO_ROOT/data"
TODAY_UTC="$(date -u +%Y-%m-%d)"

mkdir -p "$LOG_DIR" "$LOCK_DIR"

cd "$REPO_ROOT" || { echo "[run_daily_if_due] cannot cd to $REPO_ROOT" >&2; exit 2; }

run_job() {
    local job="$1"; shift || true
    local done_file="$LOCK_DIR/.daily_${job}_done_${TODAY_UTC}"
    local flock_file="$LOCK_DIR/.daily_${job}.flock"
    local log_file
    local cmd

    case "$job" in
        tcgplayer)
            log_file="$LOG_DIR/tcgplayer_scrape.log"
            cmd=("$PY" "scripts/scrapers/daily_tcgplayer_scrape.py" "--limit" "5000" "--workers" "4" "$@")
            ;;
        justtcg)
            log_file="$LOG_DIR/justtcg_batch.log"
            export JUSTTCG_API_KEY="${JUSTTCG_API_KEY:-tcg_d507de6c16dc43bdaaa29f7f4cece6cd}"
            cmd=("$PY" "scripts/scrapers/batch_justtcg.py" "--limit" "2000" "--include-jp" "$@")
            ;;
        tcgcsv_jp)
            log_file="$LOG_DIR/tcgcsv_jp_ingest.log"
            cmd=("$PY" "scripts/scrapers/tcgcsv_ingest_jp.py" "$@")
            ;;
        *)
            echo "[run_daily_if_due] unknown job: $job" >&2
            return 2
            ;;
    esac

    if [ -f "$done_file" ]; then
        # Fast no-op path. Stay silent on stdout/stderr; just timestamp
        # the log so we can see when the wrapper fired and skipped.
        printf '[%s] %s: today (%s) already complete — skipping.\n' \
            "$(date -u +%FT%TZ)" "$job" "$TODAY_UTC" >> "$log_file"
        return 0
    fi

    # Acquire non-blocking exclusive lock. If another caller holds it,
    # exit 0 quietly — they're already running today's job for us.
    exec 9>"$flock_file"
    if ! flock -n 9; then
        printf '[%s] %s: another instance is running — skipping.\n' \
            "$(date -u +%FT%TZ)" "$job" >> "$log_file"
        return 0
    fi

    # Re-check lockfile under the flock — closes the gap where two
    # callers checked simultaneously, one completed, the other arrived.
    if [ -f "$done_file" ]; then
        printf '[%s] %s: completed by sibling — skipping.\n' \
            "$(date -u +%FT%TZ)" "$job" >> "$log_file"
        return 0
    fi

    printf '\n[%s] %s: START (UTC date=%s) — cmd: %s\n' \
        "$(date -u +%FT%TZ)" "$job" "$TODAY_UTC" "${cmd[*]}" >> "$log_file"

    if "${cmd[@]}" >> "$log_file" 2>&1; then
        : > "$done_file"
        printf '[%s] %s: DONE — wrote %s\n' \
            "$(date -u +%FT%TZ)" "$job" "$done_file" >> "$log_file"
        return 0
    else
        local rc=$?
        printf '[%s] %s: FAILED (rc=%d) — NOT writing done-file, will retry on next trigger.\n' \
            "$(date -u +%FT%TZ)" "$job" "$rc" >> "$log_file"
        return "$rc"
    fi
}

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <tcgplayer|justtcg|tcgcsv_jp|all> [extra args...]" >&2
    exit 2
fi

JOB="$1"; shift

if [ "$JOB" = "all" ]; then
    rc=0
    run_job tcgplayer || rc=$?
    run_job justtcg   || rc=$?
    run_job tcgcsv_jp || rc=$?
    exit "$rc"
fi

run_job "$JOB" "$@"
