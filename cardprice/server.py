"""Minimal HTTP server for phone-based card scanning.

Start: python -m cardprice.server [--port 8888]
Then open http://<your-wsl-ip>:8888 on your phone.

Endpoints:
    GET  /           -> Mobile-friendly upload page
    POST /scan       -> Upload image, identify card, return JSON
    GET  /inventory  -> Current inventory as JSON
"""

import argparse
import json
import logging
import re
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("data/inbox")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

HTML_PAGE = """\
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Card Scanner</title>
<style>
body { font-family: -apple-system, sans-serif; max-width: 500px; margin: 20px auto; padding: 0 15px; background: #1a1a2e; color: #eee; }
h1 { text-align: center; color: #e94560; }
.upload-btn { display: block; width: 100%; padding: 20px; font-size: 18px; background: #e94560; color: white; border: none; border-radius: 12px; cursor: pointer; margin: 10px 0; }
.upload-btn:active { background: #c23152; }
input[type=file] { display: none; }
.result { background: #16213e; padding: 15px; border-radius: 8px; margin: 15px 0; display: none; }
.result.show { display: block; }
.result h3 { color: #e94560; margin: 0 0 10px; }
.price { font-size: 24px; color: #4ecca3; }
.confidence { color: #888; }
#preview { max-width: 100%; border-radius: 8px; margin: 10px 0; display: none; }
.spinner { display: none; text-align: center; padding: 20px; }
.spinner.show { display: block; }
</style>
</head>
<body>
<h1>Pokemon Card Scanner</h1>
<form id="scanForm">
    <label class="upload-btn" for="camera">Take Photo</label>
    <input type="file" id="camera" accept="image/*" capture="environment">
    <label class="upload-btn" for="gallery" style="background:#16213e;border:2px solid #e94560;">Choose from Gallery</label>
    <input type="file" id="gallery" accept="image/*">
</form>
<img id="preview">
<div class="spinner" id="spinner">Scanning...</div>
<div class="result" id="result">
    <h3 id="cardName"></h3>
    <div class="price" id="cardPrice"></div>
    <div class="confidence" id="cardConf"></div>
    <div id="cardMeta"></div>
</div>
<script>
function handleFile(file) {
    if (!file) return;
    var preview = document.getElementById('preview');
    preview.src = URL.createObjectURL(file);
    preview.style.display = 'block';
    var spinner = document.getElementById('spinner');
    var result = document.getElementById('result');
    spinner.classList.add('show');
    result.classList.remove('show');
    var fd = new FormData();
    fd.append('image', file);
    fetch('/scan', {method: 'POST', body: fd})
        .then(function(r) { return r.json(); })
        .then(function(data) {
            spinner.classList.remove('show');
            result.classList.add('show');
            document.getElementById('cardName').textContent = data.card_name || 'Unknown Card';
            document.getElementById('cardPrice').textContent = data.market_price ? '$' + data.market_price : 'No price data';
            document.getElementById('cardConf').textContent =
                (data.confidence ? (data.confidence * 100).toFixed(0) + '% confidence' : '') +
                (data.method ? ' via ' + data.method : '');
            document.getElementById('cardMeta').textContent = data.card_id || '';
        })
        .catch(function(e) {
            spinner.classList.remove('show');
            result.classList.add('show');
            document.getElementById('cardName').textContent = 'Error: ' + e;
        });
}
document.getElementById('camera').onchange = function() { handleFile(this.files[0]); };
document.getElementById('gallery').onchange = function() { handleFile(this.files[0]); };
</script>
</body>
</html>
"""


def _parse_multipart(body, content_type):
    """Extract first file from multipart/form-data body.

    Returns (filename, file_bytes) or (None, None).
    """
    m = re.search(r"boundary=([^\s;]+)", content_type)
    if not m:
        return None, None
    boundary = m.group(1).encode()
    parts = body.split(b"--" + boundary)
    for part in parts:
        if b"Content-Disposition" not in part:
            continue
        header_end = part.find(b"\r\n\r\n")
        if header_end < 0:
            continue
        header_block = part[:header_end].decode(errors="replace")
        file_data = part[header_end + 4:]
        # Strip trailing \r\n
        if file_data.endswith(b"\r\n"):
            file_data = file_data[:-2]
        fn_match = re.search(r'filename="([^"]+)"', header_block)
        if fn_match:
            return fn_match.group(1), file_data
    return None, None


class ScanHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self._send_html(HTML_PAGE)
        elif self.path == "/inventory":
            self._send_inventory()
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/scan":
            self._handle_scan()
        else:
            self.send_error(404)

    def _send_html(self, html):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode())

    def _send_json(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _handle_scan(self):
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self.send_error(400, "Expected multipart/form-data")
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        filename, file_data = _parse_multipart(body, content_type)
        if not filename or not file_data:
            self.send_error(400, "No image uploaded")
            return

        # Save uploaded image
        ext = Path(filename).suffix or ".jpg"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = UPLOAD_DIR / f"scan_{timestamp}{ext}"
        save_path.write_bytes(file_data)

        # Run identification
        try:
            from cardprice.ml import identify_card
            from cardprice.db.session import SessionLocal
            from sqlalchemy import text as sql_text

            with SessionLocal() as session:
                result = identify_card(str(save_path), session=session)

                response = {
                    "card_id": result["card_id"],
                    "confidence": result["confidence"],
                    "method": result["method"],
                    "card_name": None,
                    "market_price": None,
                    "set_name": None,
                }

                if result["card_id"]:
                    row = session.execute(
                        sql_text("""
                            SELECT c.name, s.name as set_name, p.market_price
                            FROM dim_cards c
                            JOIN dim_sets s ON s.set_id = c.set_id
                            LEFT JOIN LATERAL (
                                SELECT market_price FROM fact_market_prices
                                WHERE card_id = c.card_id
                                ORDER BY price_date DESC LIMIT 1
                            ) p ON true
                            WHERE c.card_id = :cid
                        """),
                        {"cid": result["card_id"]},
                    ).fetchone()
                    if row:
                        response["card_name"] = row.name
                        response["set_name"] = row.set_name
                        response["market_price"] = (
                            float(row.market_price) if row.market_price else None
                        )

        except Exception as e:
            logger.error("Scan error: %s", e)
            response = {"error": str(e)}

        self._send_json(response)

    def _send_inventory(self):
        try:
            from cardprice.db.session import SessionLocal
            from sqlalchemy import text as sql_text

            with SessionLocal() as session:
                rows = session.execute(sql_text("""
                    SELECT ui.card_id, dc.name, dc.set_id, ui.quantity,
                           ui.condition, lp.market_price
                    FROM user_inventory ui
                    JOIN dim_cards dc ON dc.card_id = ui.card_id
                    LEFT JOIN LATERAL (
                        SELECT market_price FROM fact_market_prices
                        WHERE card_id = ui.card_id
                        ORDER BY price_date DESC LIMIT 1
                    ) lp ON true
                    ORDER BY dc.name
                """)).fetchall()

                items = [
                    {
                        "card_id": r.card_id,
                        "name": r.name,
                        "set_id": r.set_id,
                        "quantity": r.quantity,
                        "condition": r.condition,
                        "market_price": (
                            float(r.market_price) if r.market_price else None
                        ),
                    }
                    for r in rows
                ]
                self._send_json({"items": items, "count": len(items)})
        except Exception as e:
            self._send_json({"error": str(e)})

    def log_message(self, fmt, *args):
        logger.info(fmt, *args)


def run_server(host="0.0.0.0", port=8888):
    """Start the HTTP server."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    server = HTTPServer((host, port), ScanHandler)
    print(f"Card scanner server running at http://{host}:{port}")
    print("Open this URL on your phone to scan cards")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description="Card scanner HTTP server")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
