"""Minimal HTTP server for phone-based card scanning.

Start: python -m cardprice.server [--port 8888]
Then open http://<your-wsl-ip>:8888 on your phone.

Endpoints:
    GET  /           -> Mobile-friendly upload page (with QR code)
    GET  /qr         -> QR code PNG image of the server URL
    POST /scan       -> Upload image, identify card, return JSON
    GET  /inventory  -> Current inventory as JSON
"""

import argparse
import json
import logging
import re
import socket
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("data/inbox")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB

# Server port stored at startup so QR code can encode the full URL
_server_port = 8888


def _get_lan_ip():
    """Return the LAN IP address of this machine."""
    try:
        # Connect to an external address to determine which interface is used
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _generate_qr_png(url):
    """Generate a QR code PNG as bytes using the qrcode library.

    Returns PNG bytes, or None if the library is not installed.
    """
    try:
        import qrcode
        import io
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=8,
            border=2,
        )
        qr.add_data(url)
        qr.make(fit=True)
        img = qr.make_image(fill_color="white", back_color="#1a1a2e")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except ImportError:
        return None

HTML_PAGE = r"""<!DOCTYPE html>
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
.qr-section { text-align: center; background: #16213e; border-radius: 12px; padding: 15px; margin: 0 0 15px; }
.qr-section p { margin: 5px 0 10px; color: #888; font-size: 14px; }
.qr-section .url { font-family: monospace; color: #4ecca3; font-size: 13px; word-break: break-all; }
#qrCanvas { image-rendering: pixelated; border-radius: 4px; }
</style>
</head>
<body>
<h1>Pokemon Card Scanner</h1>
<div class="qr-section" id="qrSection">
    <p>Scan QR code to open on your phone</p>
    <canvas id="qrCanvas"></canvas>
    <br>
    <span class="url" id="serverUrl"></span>
</div>
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
// Minimal QR Code generator in JS — zero external dependencies.
// Supports byte-mode encoding up to version 6 (EC level M).
var QRGen=(function(){
"use strict";
var EXP=new Array(256),LOG=new Array(256);
(function(){var v=1;for(var i=0;i<255;i++){EXP[i]=v;LOG[v]=i;v<<=1;if(v>=256)v^=0x11d;}EXP[255]=EXP[0];})();
function gfMul(a,b){return a===0||b===0?0:EXP[(LOG[a]+LOG[b])%255];}
function polyMul(a,b){var r=new Array(a.length+b.length-1).fill(0);for(var i=0;i<a.length;i++)for(var j=0;j<b.length;j++)r[i+j]^=gfMul(a[i],b[j]);return r;}
function ecBytes(data,ecLen){
    var gen=[1];for(var i=0;i<ecLen;i++)gen=polyMul(gen,[1,EXP[i]]);
    var msg=new Array(data.length+ecLen).fill(0);for(var i=0;i<data.length;i++)msg[i]=data[i];
    for(var i=0;i<data.length;i++){var c=msg[i];if(c!==0)for(var j=0;j<gen.length;j++)msg[i+j]^=gfMul(gen[j],c);}
    return msg.slice(data.length);
}
var VERSIONS=[
    null,
    {total:26,ec:10,cap:16},{total:44,ec:16,cap:28},{total:70,ec:26,cap:44},
    {total:100,ec:18,cap:82},{total:134,ec:26,cap:108},{total:172,ec:18,cap:154}
];
var ALIGN=[null,null,[6,18],[6,22],[6,26],[6,30],[6,34]];
function chooseVersion(len){for(var v=1;v<=6;v++){if(len<=VERSIONS[v].cap)return v;}return 6;}
function makeMatrix(sz){var m=[];for(var i=0;i<sz;i++){var r=[];for(var j=0;j<sz;j++)r.push(null);m.push(r);}return m;}
function addFinder(m,row,col){
    for(var r=-1;r<=7;r++)for(var c=-1;c<=7;c++){
        var rr=row+r,cc=col+c;if(rr<0||rr>=m.length||cc<0||cc>=m.length)continue;
        m[rr][cc]=((r>=0&&r<=6&&(c===0||c===6))||(c>=0&&c<=6&&(r===0||r===6))||(r>=2&&r<=4&&c>=2&&c<=4))?1:0;
    }
}
function addAlignment(m,row,col){
    for(var r=-2;r<=2;r++)for(var c=-2;c<=2;c++)
        m[row+r][col+c]=(Math.abs(r)===2||Math.abs(c)===2||(r===0&&c===0))?1:0;
}
function addTimingPatterns(m){var sz=m.length;for(var i=8;i<sz-8;i++){if(m[6][i]===null)m[6][i]=(i%2===0)?1:0;if(m[i][6]===null)m[i][6]=(i%2===0)?1:0;}}
function reserveFormatInfo(m){
    var sz=m.length;
    for(var i=0;i<8;i++){if(m[8][i]===null)m[8][i]=0;if(m[i][8]===null)m[i][8]=0;if(m[8][sz-1-i]===null)m[8][sz-1-i]=0;if(m[sz-1-i][8]===null)m[sz-1-i][8]=0;}
    if(m[8][8]===null)m[8][8]=0;m[sz-8][8]=1;
}
function placeData(m,bits){
    var sz=m.length,idx=0;
    for(var col=sz-1;col>=1;col-=2){
        if(col===6)col=5;
        for(var row=0;row<sz;row++){for(var c=0;c<2;c++){
            var cc=col-c,goUp=((Math.floor((sz-1-col)/2))%2===0),rr=goUp?(sz-1-row):row;
            if(m[rr][cc]===null){m[rr][cc]=(idx<bits.length)?bits[idx]:0;idx++;}
        }}
    }
}
function isReserved(m,r,c,sz){
    if(r<9&&c<9)return true;if(r<9&&c>=sz-8)return true;if(r>=sz-8&&c<9)return true;
    if(r===6||c===6)return true;return false;
}
function applyMask0(m,sz){for(var r=0;r<sz;r++)for(var c=0;c<sz;c++){if(!isReserved(m,r,c,sz)&&(r+c)%2===0)m[r][c]^=1;}}
function writeFormatInfo(m){
    var sz=m.length;var bits=[1,0,1,0,1,0,0,0,0,0,1,0,0,1,0]; // ECM mask0
    var hpos=[0,1,2,3,4,5,7,8];
    for(var i=0;i<8;i++)m[8][hpos[i]]=bits[i];
    for(var i=0;i<7;i++)m[8][sz-7+i]=bits[8+i];
    for(var i=0;i<8;i++)m[hpos[7-i]][8]=bits[i];
    for(var i=0;i<7;i++)m[sz-1-i][8]=bits[8+i];
}
function pushBitsTo(arr,val,len){for(var i=len-1;i>=0;i--)arr.push((val>>i)&1);}
function encode(text){
    var bytes=[];for(var i=0;i<text.length;i++){var cp=text.charCodeAt(i);if(cp<128)bytes.push(cp);else if(cp<0x800){bytes.push(0xc0|(cp>>6));bytes.push(0x80|(cp&0x3f));}else{bytes.push(0xe0|(cp>>12));bytes.push(0x80|((cp>>6)&0x3f));bytes.push(0x80|(cp&0x3f));}}
    var version=chooseVersion(bytes.length);var vi=VERSIONS[version];var sz=17+version*4;
    var dataBits=[];
    pushBitsTo(dataBits,4,4);pushBitsTo(dataBits,bytes.length,version<=9?8:16);
    for(var i=0;i<bytes.length;i++)pushBitsTo(dataBits,bytes[i],8);
    var totalBits=vi.cap*8;var tl=Math.min(4,totalBits-dataBits.length);pushBitsTo(dataBits,0,tl);
    while(dataBits.length%8!==0)dataBits.push(0);
    var pad=[0xEC,0x11],pi=0;while(dataBits.length<totalBits){pushBitsTo(dataBits,pad[pi],8);pi^=1;}
    var dataBytes=[];for(var i=0;i<dataBits.length;i+=8){var b=0;for(var j=0;j<8;j++)b=(b<<1)|dataBits[i+j];dataBytes.push(b);}
    var ecCW=ecBytes(dataBytes,vi.ec);
    var allBits=[];for(var i=0;i<dataBytes.length;i++)pushBitsTo(allBits,dataBytes[i],8);
    for(var i=0;i<ecCW.length;i++)pushBitsTo(allBits,ecCW[i],8);
    var m=makeMatrix(sz);addFinder(m,0,0);addFinder(m,0,sz-7);addFinder(m,sz-7,0);
    if(ALIGN[version]){var ap=ALIGN[version];for(var i=0;i<ap.length;i++)for(var j=0;j<ap.length;j++){if(i===0&&j===0)continue;if(i===0&&j===ap.length-1)continue;if(i===ap.length-1&&j===0)continue;addAlignment(m,ap[i],ap[j]);}}
    addTimingPatterns(m);reserveFormatInfo(m);placeData(m,allBits);applyMask0(m,sz);writeFormatInfo(m);
    return m;
}
return{encode:encode};
})();

function drawQR(canvasId,text,cellSize){
    cellSize=cellSize||6;var matrix=QRGen.encode(text);var sz=matrix.length;
    var canvas=document.getElementById(canvasId);canvas.width=sz*cellSize;canvas.height=sz*cellSize;
    var ctx=canvas.getContext('2d');ctx.fillStyle='#1a1a2e';ctx.fillRect(0,0,canvas.width,canvas.height);
    ctx.fillStyle='#ffffff';
    for(var r=0;r<sz;r++)for(var c=0;c<sz;c++)if(matrix[r][c])ctx.fillRect(c*cellSize,r*cellSize,cellSize,cellSize);
}
// Render QR code with the server URL; hide on mobile (already on the phone)
(function(){
    var url='http://'+location.host;
    document.getElementById('serverUrl').textContent=url;
    drawQR('qrCanvas',url,6);
    if(/Mobi|Android|iPhone|iPad/i.test(navigator.userAgent))
        document.getElementById('qrSection').style.display='none';
})();
</script>
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
    m = re.search(r'boundary="?([^\s";]+)"?', content_type)
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
        # Strip trailing CRLF and closing boundary marker (--)
        if file_data.endswith(b"--\r\n"):
            file_data = file_data[:-4]
        if file_data.endswith(b"\r\n"):
            file_data = file_data[:-2]
        fn_match = re.search(r'filename="([^"]*)"', header_block)
        if fn_match and fn_match.group(1):
            return fn_match.group(1), file_data
    return None, None


class ScanHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self._send_html(HTML_PAGE)
        elif self.path == "/qr":
            self._send_qr()
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

    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _send_qr(self):
        """Serve QR code PNG for the server URL (requires qrcode library)."""
        url = f"http://{_get_lan_ip()}:{_server_port}"
        png_data = _generate_qr_png(url)
        if png_data:
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(png_data)))
            self.end_headers()
            self.wfile.write(png_data)
        else:
            # qrcode library not installed — return a helpful message
            self._send_json(
                {"error": "qrcode library not installed; QR is rendered client-side via JS"},
                status=501,
            )

    def _handle_scan(self):
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self.send_error(400, "Expected multipart/form-data")
            return

        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            self.send_error(411, "Content-Length required")
            return
        try:
            length = int(raw_length)
        except (ValueError, TypeError):
            self.send_error(400, "Invalid Content-Length")
            return
        if length <= 0:
            self.send_error(400, "Empty request body")
            return
        if length > MAX_UPLOAD_BYTES:
            self.send_error(413, "Upload too large (max 20 MB)")
            return

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

            self._send_json(response)

        except Exception as e:
            logger.error("Scan error: %s", e)
            self._send_json({"error": str(e)}, status=500)

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
            logger.error("Inventory error: %s", e)
            self._send_json({"error": str(e)}, status=500)

    def log_message(self, fmt, *args):
        logger.info(fmt, *args)


def run_server(host="0.0.0.0", port=8888):
    """Start the HTTP server."""
    global _server_port
    _server_port = port
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    server = HTTPServer((host, port), ScanHandler)
    lan_ip = _get_lan_ip()
    print(f"Card scanner server running at http://{host}:{port}")
    print(f"LAN URL (for phone): http://{lan_ip}:{port}")
    print("Open this URL on your phone, or scan the QR code on the landing page")
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
