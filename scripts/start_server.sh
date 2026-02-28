#!/bin/bash
# Start the card scanner HTTP server with network info

PORT="${1:-8888}"
IP=$(hostname -I | awk '{print $1}')

echo "========================================"
echo "  Pokemon Card Scanner"
echo "========================================"
echo ""
echo "  Local:   http://localhost:${PORT}"
echo "  Network: http://${IP}:${PORT}"
echo ""
echo "  Open the Network URL on your phone"
echo "  to scan cards with your camera."
echo "========================================"
echo ""

cd "$(dirname "$0")/.." || exit 1
exec python -m cardprice.cli server --port "$PORT"
