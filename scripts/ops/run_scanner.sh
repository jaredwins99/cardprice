#!/bin/bash
#
# run_scanner.sh: Keeps the card scanner server running with auto-restart on crash.
# Also starts inotifywait to monitor data/pending_scans/ for new pending scan files.
#
# Usage:
#   ./scripts/run_scanner.sh [port]
#
# Default port: 8888
#

set -e

PORT="${1:-8888}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PENDING_DIR="$PROJECT_DIR/data/pending_scans"

# Ensure pending_scans directory exists
mkdir -p "$PENDING_DIR"

# PIDs for background processes
SERVER_PID=""
WATCH_PID=""

# Cleanup function: kill both processes on exit
cleanup() {
    echo "Shutting down..."
    if [ -n "$SERVER_PID" ]; then
        kill "$SERVER_PID" 2>/dev/null || true
    fi
    if [ -n "$WATCH_PID" ]; then
        kill "$WATCH_PID" 2>/dev/null || true
    fi
    exit 0
}

# Install signal handler for Ctrl+C and termination
trap cleanup SIGINT SIGTERM

# Determine LAN IP for QR code
LAN_IP=$(python3 -c "
import socket
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0.5)
    s.connect(('8.8.8.8', 80))
    print(s.getsockname()[0])
    s.close()
except:
    print('127.0.0.1')
" 2>/dev/null || echo "127.0.0.1")

# Start inotifywait watcher in background
# Watches for new JSON files in data/pending_scans/
echo "Starting inotifywait watcher on $PENDING_DIR..."
(
    inotifywait -m -e create -e moved_to --format '%f' "$PENDING_DIR" |
    while read filename; do
        if [[ "$filename" == *.json ]]; then
            echo "[$(date +'%Y-%m-%d %H:%M:%S')] New pending scan: $filename"
        fi
    done
) &
WATCH_PID=$!

# Server restart loop
echo "Starting card scanner server..."
echo "LAN URL: http://$LAN_IP:$PORT"
echo "QR info: Scan on landing page to get full address"
echo ""
echo "Server will auto-restart on crash. Press Ctrl+C to exit."
echo ""

while true; do
    cd "$PROJECT_DIR"
    python -m cardprice.server --port "$PORT" &
    SERVER_PID=$!

    # Wait for server process; if it exits, loop will restart it
    wait $SERVER_PID

    # Server crashed, wait a moment before restarting
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Server crashed, restarting in 3 seconds..."
    sleep 3
done
