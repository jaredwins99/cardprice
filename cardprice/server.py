"""Minimal HTTP server for phone-based card scanning.

Start: python -m cardprice.server [--port 8888]
Then open http://<your-wsl-ip>:8888 on your phone.

Endpoints:
    GET  /           -> Mobile-friendly upload page (with QR code)
    GET  /qr         -> QR code PNG image of the server URL
    POST /scan       -> Upload image, identify card, return JSON
    POST /scan-url   -> Download image from URL, identify card, return JSON
    POST /scan-page  -> Upload binder page photo, segment & identify cards
    GET  /pending    -> List pending scans awaiting identification
    GET  /history    -> Last 50 scans (resolved + pending) sorted by timestamp desc
    GET  /stats      -> Scanning statistics (counts, methods, confidence, index sizes)
    GET  /price-history/<card_id> -> Last 30 days of market prices as JSON array
    GET  /events/<scan_id> -> SSE stream for scan result updates
    POST /resolve    -> Resolve a pending scan with correct card_id
    POST /resolve-batch -> Resolve multiple pending scans at once
    POST /inventory/add -> Add card to inventory (upsert)
    POST /inventory/remove -> Remove card from inventory (decrement)
    GET  /inventory  -> Current inventory as JSON
    GET  /export     -> Export inventory as CSV attachment
    GET  /card-image/<card_id> -> Serve local card reference image (PNG)
"""

import argparse
import csv
import io
import json
import logging
import os
import re
import socket
import time
import urllib.request
import urllib.error
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("data/inbox")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

PENDING_DIR = Path("data/pending_scans")
PENDING_DIR.mkdir(parents=True, exist_ok=True)

CARD_IMAGES_DIR = Path("data/card_images")

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB

# Duplicate detection: max Hamming distance to consider a duplicate
_DEDUP_HASH_THRESHOLD = 3

# Server port stored at startup so QR code can encode the full URL
_server_port = 8888


def _compute_phash(image_path):
    """Compute perceptual hash of an image, returning hex string or None."""
    try:
        import imagehash
        from PIL import Image
        img = Image.open(image_path)
        return str(imagehash.phash(img))
    except Exception as e:
        logger.warning("Failed to compute phash for %s: %s", image_path, e)
        return None


def _find_duplicate_scan(phash_hex):
    """Check all scans in pending_scans/ for a matching phash.

    Returns the cached scan dict if a duplicate is found (Hamming distance
    < _DEDUP_HASH_THRESHOLD), otherwise None.
    """
    if not phash_hex:
        return None
    try:
        import imagehash
        query_hash = imagehash.hex_to_hash(phash_hex)
    except Exception:
        return None

    for f in PENDING_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        stored_hex = data.get("phash")
        if not stored_hex:
            continue
        try:
            stored_hash = imagehash.hex_to_hash(stored_hex)
            distance = query_hash - stored_hash
            if distance < _DEDUP_HASH_THRESHOLD:
                logger.info(
                    "Duplicate scan detected (distance=%d): %s matches %s",
                    distance, phash_hex, f.name,
                )
                return data
        except Exception:
            continue
    return None


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
.toggle-row { display: flex; align-items: center; justify-content: space-between; background: #16213e; padding: 12px 15px; border-radius: 8px; margin: 10px 0; }
.toggle-row label { color: #ccc; font-size: 15px; }
.toggle-switch { position: relative; width: 50px; height: 28px; }
.toggle-switch input { opacity: 0; width: 0; height: 0; }
.toggle-slider { position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background: #333; border-radius: 28px; transition: 0.3s; }
.toggle-slider:before { content: ""; position: absolute; height: 22px; width: 22px; left: 3px; bottom: 3px; background: #eee; border-radius: 50%; transition: 0.3s; }
.toggle-switch input:checked + .toggle-slider { background: #4ecca3; }
.toggle-switch input:checked + .toggle-slider:before { transform: translateX(22px); }
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
<div class="toggle-row">
    <label for="continuousScan">Continuous scan mode</label>
    <div class="toggle-switch">
        <input type="checkbox" id="continuousScan">
        <span class="toggle-slider"></span>
    </div>
</div>
<form id="pageForm">
    <label class="upload-btn" for="pageCamera" style="background:#4ecca3;color:#1a1a2e;margin-top:20px;">Scan Binder Page</label>
    <input type="file" id="pageCamera" accept="image/*" capture="environment">
    <label class="upload-btn" for="pageGallery" style="background:#16213e;border:2px solid #4ecca3;color:#4ecca3;">Page from Gallery</label>
    <input type="file" id="pageGallery" accept="image/*">
</form>
<img id="preview">
<div class="spinner" id="spinner">Scanning...</div>
<div class="result" id="result">
    <h3 id="cardName"></h3>
    <div class="price" id="cardPrice"></div>
    <div class="confidence" id="cardConf"></div>
    <div id="cardMeta"></div>
    <img id="refImage" style="display:none;max-width:200px;margin:12px auto;border-radius:8px;box-shadow:0 2px 12px rgba(0,0,0,0.5)" />
    <canvas id="sparkline" width="150" height="40" style="display:none;margin:10px 0;"></canvas>
    <div id="sparkLabel" style="display:none;font-size:11px;color:#888;"></div>
    <button id="addInventoryBtn" style="display:none;margin:12px auto;padding:10px 24px;background:#4ecca3;color:#1a1a2e;border:none;border-radius:8px;font-size:16px;font-weight:bold;cursor:pointer;" onclick="addToInventory()">Add to Inventory</button>
    <div id="inventoryMsg" style="display:none;font-size:13px;margin-top:6px;"></div>
</div>
<div class="result" id="pageResult">
    <h3 style="color:#4ecca3;">Binder Page Results</h3>
    <div id="pageTotal" class="price"></div>
    <div id="pageCards"></div>
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
            if (data.status === 'pending') {
                document.getElementById('cardName').textContent = 'Queued for identification...';
                document.getElementById('cardPrice').textContent = 'Checking every 3s';
                document.getElementById('cardConf').textContent = '';
                document.getElementById('cardMeta').textContent = '';
                pollResult(data.scan_id);
            } else {
                showResult(data);
            }
        })
        .catch(function(e) {
            spinner.classList.remove('show');
            result.classList.add('show');
            document.getElementById('cardName').textContent = 'Error: ' + e;
        });
}
function showResult(data) {
    document.getElementById('cardName').textContent = data.card_name || 'Unknown Card';
    document.getElementById('cardPrice').textContent = data.market_price ? '$' + data.market_price : 'No price data';
    document.getElementById('cardConf').textContent =
        (data.confidence ? (data.confidence * 100).toFixed(0) + '% confidence' : '') +
        (data.method ? ' via ' + data.method : '');
    document.getElementById('cardMeta').textContent = data.card_id || '';
    var refImg = document.getElementById('refImage');
    if (data.image_url) {
        refImg.src = data.image_url;
        refImg.style.display = 'block';
    } else {
        refImg.style.display = 'none';
    }
    // Sparkline: fetch 30-day price history and draw
    var spark = document.getElementById('sparkline');
    var sparkLabel = document.getElementById('sparkLabel');
    spark.style.display = 'none';
    sparkLabel.style.display = 'none';
    if (data.card_id) {
        fetch('/price-history/' + encodeURIComponent(data.card_id))
            .then(function(r) { return r.json(); })
            .then(function(pts) {
                if (!pts || pts.length < 2) return;
                // pts are newest-first from API; reverse to chronological order
                pts = pts.slice().reverse();
                var prices = pts.map(function(p) { return p.price; });
                var minP = Math.min.apply(null, prices);
                var maxP = Math.max.apply(null, prices);
                var range = maxP - minP || 1;
                var W = 150, H = 40, pad = 2;
                spark.width = W; spark.height = H;
                spark.style.display = 'block';
                var ctx = spark.getContext('2d');
                ctx.clearRect(0, 0, W, H);
                var up = prices[prices.length - 1] >= prices[0];
                ctx.strokeStyle = up ? '#4ecca3' : '#e94560';
                ctx.lineWidth = 1.5;
                ctx.beginPath();
                for (var i = 0; i < prices.length; i++) {
                    var x = pad + (i / (prices.length - 1)) * (W - 2 * pad);
                    var y = H - pad - ((prices[i] - minP) / range) * (H - 2 * pad);
                    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
                }
                ctx.stroke();
                sparkLabel.style.display = 'block';
                var diff = prices[prices.length - 1] - prices[0];
                var sign = diff >= 0 ? '+' : '';
                sparkLabel.textContent = '30d: ' + sign + diff.toFixed(2) + ' (' + pts[0].date + ' \u2192 ' + pts[pts.length - 1].date + ')';
                sparkLabel.style.color = up ? '#4ecca3' : '#e94560';
            })
            .catch(function() {});
    }
    // Show/hide Add to Inventory button
    var addBtn = document.getElementById('addInventoryBtn');
    var invMsg = document.getElementById('inventoryMsg');
    invMsg.style.display = 'none';
    invMsg.textContent = '';
    if (data.card_id) {
        addBtn.style.display = 'block';
        addBtn.dataset.cardId = data.card_id;
    } else {
        addBtn.style.display = 'none';
    }
    reopenCamera();
}
function addToInventory() {
    var btn = document.getElementById('addInventoryBtn');
    var msg = document.getElementById('inventoryMsg');
    var cardId = btn.dataset.cardId;
    if (!cardId) return;
    btn.disabled = true;
    btn.textContent = 'Adding...';
    fetch('/inventory/add', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({card_id: cardId, quantity: 1})
    })
    .then(function(r) { return r.json(); })
    .then(function(data) {
        btn.disabled = false;
        btn.textContent = 'Add to Inventory';
        msg.style.display = 'block';
        if (data.error) {
            msg.style.color = '#e94560';
            msg.textContent = data.error;
        } else {
            msg.style.color = '#4ecca3';
            msg.textContent = 'Added! Total in inventory: ' + data.quantity;
        }
    })
    .catch(function(e) {
        btn.disabled = false;
        btn.textContent = 'Add to Inventory';
        msg.style.display = 'block';
        msg.style.color = '#e94560';
        msg.textContent = 'Error: ' + e;
    });
}
function pollResult(scanId) {
    if (typeof EventSource !== 'undefined') {
        var es = new EventSource('/events/' + scanId);
        es.addEventListener('resolved', function(e) {
            es.close();
            showResult(JSON.parse(e.data));
        });
        es.addEventListener('timeout', function() {
            es.close();
            document.getElementById('cardName').textContent = 'Identification timed out';
            document.getElementById('cardPrice').textContent = '';
        });
        es.onerror = function() {
            es.close();
            // Fallback to polling on SSE failure
            _pollFallback(scanId);
        };
    } else {
        _pollFallback(scanId);
    }
}
function _pollFallback(scanId) {
    var poll = setInterval(function() {
        fetch('/result/' + scanId)
            .then(function(r) { return r.json(); })
            .then(function(data) {
                if (data.status === 'resolved') {
                    clearInterval(poll);
                    showResult(data);
                }
            });
    }, 3000);
}
document.getElementById('camera').onchange = function() { handleFile(this.files[0]); this.value=''; };
document.getElementById('gallery').onchange = function() { handleFile(this.files[0]); this.value=''; };
function reopenCamera() {
    if (document.getElementById('continuousScan').checked) {
        setTimeout(function() { document.getElementById('camera').click(); }, 1200);
    }
}
function handlePageFile(file) {
    if (!file) return;
    var preview = document.getElementById('preview');
    preview.src = URL.createObjectURL(file);
    preview.style.display = 'block';
    var spinner = document.getElementById('spinner');
    var pageResult = document.getElementById('pageResult');
    var result = document.getElementById('result');
    spinner.classList.add('show');
    pageResult.classList.remove('show');
    result.classList.remove('show');
    var fd = new FormData();
    fd.append('image', file);
    fetch('/scan-page', {method: 'POST', body: fd})
        .then(function(r) { return r.json(); })
        .then(function(data) {
            spinner.classList.remove('show');
            pageResult.classList.add('show');
            if (data.error) {
                document.getElementById('pageTotal').textContent = 'Error';
                document.getElementById('pageCards').innerHTML = '<div style="color:#e94560;padding:16px;">' + data.error + '</div>';
                return;
            }
            var cards = data.cards || [];
            var total = 0;
            if (data.status === 'pending') {
                document.getElementById('pageTotal').textContent = 'Page queued for processing (' + (data.scan_id || '') + ')';
                document.getElementById('pageCards').innerHTML = '<div style="color:#888;">Segmentation unavailable. Full page image saved for later processing.</div>';
                return;
            }
            // Build grid of reference images
            var numCols = 3;
            var numRows = Math.ceil(cards.length / numCols);
            var html = '<div style="display:grid;grid-template-columns:repeat(' + numCols + ',1fr);gap:8px;max-width:600px;margin:12px auto;">';
            for (var i = 0; i < cards.length; i++) {
                var c = cards[i];
                var price = c.market_price ? parseFloat(c.market_price) : 0;
                total += price;
                var imgSrc = c.local_image_url || c.image_url || '';
                html += '<div style="background:#0f3460;border-radius:8px;overflow:hidden;text-align:center;position:relative;">';
                if (imgSrc) {
                    html += '<img src="' + imgSrc + '" style="width:100%;display:block;border-radius:8px 8px 0 0;" />';
                } else {
                    html += '<div style="width:100%;aspect-ratio:5/7;background:#16213e;display:flex;align-items:center;justify-content:center;color:#666;font-size:12px;border-radius:8px 8px 0 0;">No image</div>';
                }
                html += '<div style="padding:6px 4px;">';
                html += '<div style="font-size:12px;font-weight:bold;color:#e0e0e0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">' + (c.card_name || 'Unknown') + '</div>';
                if (c.market_price) {
                    html += '<div style="font-size:16px;font-weight:bold;color:#4ecca3;">$' + parseFloat(c.market_price).toFixed(2) + '</div>';
                } else {
                    html += '<div style="font-size:13px;color:#666;">No price</div>';
                }
                html += '<div style="font-size:10px;color:#888;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">' + (c.set_name || '') + '</div>';
                html += '</div></div>';
            }
            html += '</div>';
            document.getElementById('pageTotal').textContent = cards.length + ' cards — Total: $' + total.toFixed(2);
            document.getElementById('pageCards').innerHTML = html || '<div style="color:#888">No cards identified</div>';
        })
        .catch(function(e) {
            spinner.classList.remove('show');
            pageResult.classList.add('show');
            document.getElementById('pageTotal').textContent = 'Error';
            document.getElementById('pageCards').textContent = '' + e;
        });
}
document.getElementById('pageCamera').onchange = function() { handlePageFile(this.files[0]); };
document.getElementById('pageGallery').onchange = function() { handlePageFile(this.files[0]); };
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


def _local_image_url(card_id):
    """Return local /card-image/ URL for a card_id if the image file exists.

    card_id format: "ex14-94/normal" or "bw5-107" (with or without variant).
    Checks for normal variant PNG on disk.  Returns None if not found.
    """
    if not card_id:
        return None
    # Strip variant suffix if present (e.g. "ex14-94/normal" -> "ex14-94")
    base_id = card_id.split("/")[0] if "/" in card_id else card_id
    # Extract set_id: everything before the last '-' (e.g. "ecard3-H32" -> "ecard3")
    last_dash = base_id.rfind("-")
    if last_dash <= 0:
        return None
    set_id = base_id[:last_dash]
    image_path = CARD_IMAGES_DIR / set_id / f"{base_id}_normal.png"
    if image_path.is_file():
        return f"/card-image/{base_id}/normal"
    return None


class ScanHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self._send_html(HTML_PAGE)
        elif self.path == "/multi":
            from cardprice.scanner_ui import MULTI_CARD_HTML
            self._send_html(MULTI_CARD_HTML)
        elif self.path == "/qr":
            self._send_qr()
        elif self.path == "/inventory":
            self._send_inventory()
        elif self.path == "/export":
            self._send_csv_export()
        elif self.path == "/pending":
            self._send_pending()
        elif self.path == "/history":
            self._send_history()
        elif self.path == "/stats":
            self._send_stats()
        elif self.path.startswith("/result/"):
            self._send_result(self.path.split("/result/", 1)[1])
        elif self.path.startswith("/events/"):
            self._stream_sse(self.path.split("/events/", 1)[1])
        elif self.path.startswith("/price-history/"):
            from urllib.parse import unquote
            card_id = unquote(self.path.split("/price-history/", 1)[1])
            self._send_price_history(card_id)
        elif self.path.startswith("/card-image/"):
            from urllib.parse import unquote
            card_path = unquote(self.path.split("/card-image/", 1)[1])
            self._send_card_image(card_path)
        elif self.path.startswith("/segment-image/"):
            from urllib.parse import unquote
            seg_path = unquote(self.path.split("/segment-image/", 1)[1])
            self._send_segment_image(seg_path)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/scan":
            self._handle_scan()
        elif self.path == "/scan-url":
            self._handle_scan_url()
        elif self.path == "/scan-page":
            self._handle_scan_page()
        elif self.path == "/resolve":
            self._handle_resolve()
        elif self.path == "/resolve-batch":
            self._handle_resolve_batch()
        elif self.path == "/inventory/add":
            self._handle_inventory_add()
        elif self.path == "/inventory/remove":
            self._handle_inventory_remove()
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

        # Duplicate detection: compute phash and check for previous scan
        phash_hex = _compute_phash(str(save_path))
        if phash_hex:
            cached = _find_duplicate_scan(phash_hex)
            if cached:
                logger.info("Returning cached result for duplicate scan")
                cached_response = dict(cached)
                cached_response["duplicate"] = True
                self._send_json(cached_response)
                return

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
                    "image_url": None,
                    "local_image_url": _local_image_url(result["card_id"]),
                    "phash": phash_hex,
                }

                if result["card_id"]:
                    row = session.execute(
                        sql_text("""
                            SELECT c.name, s.name as set_name, c.image_small, p.market_price
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
                        response["image_url"] = row.image_small
                else:
                    # No confident ML match — queue for Claude Code identification
                    scan_id = timestamp
                    pending_meta = PENDING_DIR / f"{scan_id}.json"
                    pending_meta.write_text(json.dumps({
                        "scan_id": scan_id,
                        "image_path": str(save_path),
                        "status": "pending",
                        "phash": phash_hex,
                        "ml_response": result.get("raw_response", {}),
                    }))
                    response["status"] = "pending"
                    response["scan_id"] = scan_id
                    response["message"] = "Card queued for identification"

            # Save resolved scan metadata so future dedup checks can find it
            if response.get("card_id") and phash_hex:
                resolved_meta = PENDING_DIR / f"{timestamp}.json"
                if not resolved_meta.exists():
                    resolved_meta.write_text(json.dumps({
                        "scan_id": timestamp,
                        "image_path": str(save_path),
                        "status": "resolved",
                        "phash": phash_hex,
                        "card_id": response["card_id"],
                        "confidence": response["confidence"],
                        "method": response["method"],
                        "card_name": response.get("card_name"),
                        "market_price": response.get("market_price"),
                        "set_name": response.get("set_name"),
                        "image_url": response.get("image_url"),
                    }))

            self._send_json(response)

        except Exception as e:
            logger.error("Scan error: %s", e)
            self._send_json({"error": str(e)}, status=500)


    def _read_json_body(self, max_bytes=1024 * 1024):
        """Read and parse a JSON request body.  Returns dict or None (sends error)."""
        content_type = self.headers.get("Content-Type", "")
        if "application/json" not in content_type:
            self.send_error(400, "Expected application/json")
            return None
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            self.send_error(411, "Content-Length required")
            return None
        try:
            length = int(raw_length)
        except (ValueError, TypeError):
            self.send_error(400, "Invalid Content-Length")
            return None
        if length <= 0:
            self.send_error(400, "Empty request body")
            return None
        if length > max_bytes:
            self.send_error(413, "Request body too large")
            return None
        body = self.rfile.read(length)
        try:
            return json.loads(body.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            self.send_error(400, "Invalid JSON")
            return None

    def _resolve_single(self, scan_id, card_id, confidence=0.95):
        """Resolve a single pending scan.  Returns response dict or error dict."""
        pending_file = PENDING_DIR / f"{scan_id}.json"
        if not pending_file.exists():
            return {"error": f"Scan {scan_id} not found", "_status": 404}

        try:
            scan_data = json.loads(pending_file.read_text())
        except (json.JSONDecodeError, OSError) as e:
            return {"error": f"Failed to read scan data: {e}", "_status": 500}

        # Look up card info from database
        card_name = None
        set_name = None
        market_price = None
        image_url = None
        try:
            from cardprice.db.session import SessionLocal
            from sqlalchemy import text as sql_text

            with SessionLocal() as session:
                row = session.execute(
                    sql_text("""
                        SELECT c.name, s.name as set_name, c.image_small, p.market_price
                        FROM dim_cards c
                        JOIN dim_sets s ON s.set_id = c.set_id
                        LEFT JOIN LATERAL (
                            SELECT market_price FROM fact_market_prices
                            WHERE card_id = c.card_id
                            ORDER BY price_date DESC LIMIT 1
                        ) p ON true
                        WHERE c.card_id = :cid
                    """),
                    {"cid": card_id},
                ).fetchone()
                if row:
                    card_name = row.name
                    set_name = row.set_name
                    market_price = float(row.market_price) if row.market_price else None
                    image_url = row.image_small
        except Exception as e:
            logger.warning("DB lookup failed during resolve for %s: %s", card_id, e)

        # Update the pending scan file
        scan_data["status"] = "resolved"
        scan_data["card_id"] = card_id
        scan_data["confidence"] = confidence
        scan_data["method"] = "manual"
        scan_data["card_name"] = card_name
        scan_data["set_name"] = set_name
        scan_data["market_price"] = market_price
        scan_data["image_url"] = image_url
        pending_file.write_text(json.dumps(scan_data))

        return {
            "scan_id": scan_id,
            "status": "resolved",
            "card_id": card_id,
            "confidence": confidence,
            "method": "manual",
            "card_name": card_name,
            "set_name": set_name,
            "market_price": market_price,
            "image_url": image_url,
            "local_image_url": _local_image_url(card_id),
        }

    def _handle_resolve(self):
        """Resolve a single pending/unknown scan by providing the correct card_id.

        Accepts JSON: {"scan_id": "...", "card_id": "...", "confidence": 0.95}
        Updates the pending scan JSON and returns card info.
        """
        data = self._read_json_body()
        if data is None:
            return

        scan_id = data.get("scan_id")
        card_id = data.get("card_id")
        if not scan_id or not card_id:
            self._send_json({"error": "Missing required fields: scan_id, card_id"}, status=400)
            return

        confidence = data.get("confidence", 0.95)
        result = self._resolve_single(scan_id, card_id, confidence)
        status = result.pop("_status", 200)
        self._send_json(result, status=status)

    def _handle_resolve_batch(self):
        """Resolve multiple pending scans at once.

        Accepts JSON: {"resolutions": [{"scan_id": "...", "card_id": "...", "confidence": 0.95}, ...]}
        Returns results for each resolution.
        """
        data = self._read_json_body()
        if data is None:
            return

        resolutions = data.get("resolutions")
        if not resolutions or not isinstance(resolutions, list):
            self._send_json({"error": "Missing or invalid 'resolutions' array"}, status=400)
            return

        results = []
        for item in resolutions:
            scan_id = item.get("scan_id")
            card_id = item.get("card_id")
            if not scan_id or not card_id:
                results.append({"error": "Missing scan_id or card_id", "scan_id": scan_id})
                continue
            confidence = item.get("confidence", 0.95)
            result = self._resolve_single(scan_id, card_id, confidence)
            result.pop("_status", None)
            results.append(result)

        self._send_json({"results": results, "count": len(results)})

    def _handle_scan_url(self):
        """Handle image download from URL: fetch the image, identify card, return JSON.

        Accepts JSON: {"url": "https://example.com/card.jpg"}
        Downloads with 10-second timeout and 20MB size limit.
        Returns same response format as /scan.
        """
        content_type = self.headers.get("Content-Type", "")
        if "application/json" not in content_type:
            self.send_error(400, "Expected application/json")
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
        if length > 1024 * 1024:  # 1 MB max for JSON request itself
            self.send_error(413, "Request body too large")
            return

        body = self.rfile.read(length)
        try:
            request_data = json.loads(body.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            self.send_error(400, "Invalid JSON")
            return

        url = request_data.get("url")
        if not url:
            self.send_error(400, "Missing 'url' field in JSON body")
            return

        # Validate URL (basic check)
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            self.send_error(400, "Invalid URL: must start with http:// or https://")
            return

        # Download the image with timeout and size limit
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                file_data = b""
                # Read in chunks to enforce size limit
                while True:
                    chunk = response.read(1024 * 1024)  # 1 MB chunks
                    if not chunk:
                        break
                    file_data += chunk
                    if len(file_data) > MAX_UPLOAD_BYTES:
                        self.send_error(413, f"Downloaded image too large (max {MAX_UPLOAD_BYTES / 1024 / 1024:.0f} MB)")
                        return

            if not file_data:
                self.send_error(400, "Downloaded image is empty")
                return

        except urllib.error.URLError as e:
            logger.warning("URL fetch failed: %s", e)
            self.send_error(400, f"Failed to download image: {str(e)[:100]}")
            return
        except socket.timeout:
            self.send_error(408, "Download timeout (10s)")
            return
        except Exception as e:
            logger.warning("URL download error: %s", e)
            self.send_error(400, f"Download error: {str(e)[:100]}")
            return

        # Infer file extension from URL or default to .jpg
        ext = ".jpg"
        if "?" in url:
            path_part = url.split("?")[0]
        else:
            path_part = url
        if "." in path_part:
            ext_candidate = "." + path_part.split(".")[-1].lower()
            if ext_candidate in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"):
                ext = ext_candidate

        # Save downloaded image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = UPLOAD_DIR / f"scan_{timestamp}{ext}"
        save_path.write_bytes(file_data)
        logger.info("Downloaded image saved: %s (%d bytes from %s)", save_path, len(file_data), url)

        # Duplicate detection: compute phash and check for previous scan
        phash_hex = _compute_phash(str(save_path))
        if phash_hex:
            cached = _find_duplicate_scan(phash_hex)
            if cached:
                logger.info("Returning cached result for duplicate scan from URL")
                cached_response = dict(cached)
                cached_response["duplicate"] = True
                self._send_json(cached_response)
                return

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
                    "image_url": None,
                    "local_image_url": _local_image_url(result["card_id"]),
                    "phash": phash_hex,
                    "source_url": url,
                }

                if result["card_id"]:
                    row = session.execute(
                        sql_text("""
                            SELECT c.name, s.name as set_name, c.image_small, p.market_price
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
                        response["image_url"] = row.image_small
                else:
                    # No confident ML match — queue for Claude Code identification
                    scan_id = timestamp
                    pending_meta = PENDING_DIR / f"{scan_id}.json"
                    pending_meta.write_text(json.dumps({
                        "scan_id": scan_id,
                        "image_path": str(save_path),
                        "status": "pending",
                        "phash": phash_hex,
                        "source_url": url,
                        "ml_response": result.get("raw_response", {}),
                    }))
                    response["status"] = "pending"
                    response["scan_id"] = scan_id
                    response["message"] = "Card queued for identification"

            # Save resolved scan metadata so future dedup checks can find it
            if response.get("card_id") and phash_hex:
                resolved_meta = PENDING_DIR / f"{timestamp}.json"
                if not resolved_meta.exists():
                    resolved_meta.write_text(json.dumps({
                        "scan_id": timestamp,
                        "image_path": str(save_path),
                        "status": "resolved",
                        "phash": phash_hex,
                        "card_id": response["card_id"],
                        "confidence": response["confidence"],
                        "method": response["method"],
                        "card_name": response.get("card_name"),
                        "market_price": response.get("market_price"),
                        "set_name": response.get("set_name"),
                        "image_url": response.get("image_url"),
                        "source_url": url,
                    }))

            self._send_json(response)

        except Exception as e:
            logger.error("Scan-URL error: %s", e)
            self._send_json({"error": str(e)}, status=500)

    def _handle_scan_page(self):
        """Handle binder page upload: segment into individual cards, identify each."""
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

        # Save uploaded page image
        ext = Path(filename).suffix or ".jpg"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = UPLOAD_DIR / f"page_{timestamp}{ext}"
        save_path.write_bytes(file_data)
        logger.info("Binder page saved: %s (%d bytes)", save_path, len(file_data))

        # Try to segment cards from the page
        card_images = []
        segmentation_ok = False
        try:
            from cardprice.ml.card_segmenter import segment_cards
            card_images = segment_cards(str(save_path))
            segmentation_ok = True
            logger.info("Segmented %d cards from binder page", len(card_images))
        except ImportError:
            logger.info("card_segmenter not available, queuing whole page")
        except Exception as e:
            logger.warning("Segmentation failed: %s, queuing whole page", e)

        if not segmentation_ok or not card_images:
            # Queue the whole page image for later processing
            scan_id = f"page_{timestamp}"
            pending_meta = PENDING_DIR / f"{scan_id}.json"
            pending_meta.write_text(json.dumps({
                "scan_id": scan_id,
                "image_path": str(save_path),
                "status": "pending",
                "type": "binder_page",
            }))
            self._send_json({
                "status": "pending",
                "scan_id": scan_id,
                "message": "Binder page queued for processing",
                "cards": [],
            })
            return

        # Identify each segmented card
        cards = []
        try:
            from cardprice.ml import identify_page_vision_first as identify_page
            from cardprice.db.session import SessionLocal
            from sqlalchemy import text as sql_text

            with SessionLocal() as session:
                page_results = identify_page(card_images, session=session)
                for idx, (card_img_path, result) in enumerate(zip(card_images, page_results)):
                    # Compute grid position (assume 3 columns for binder pages)
                    num_cols = 3
                    row = idx // num_cols
                    col = idx % num_cols
                    # Build URL for the segmented card image
                    seg_rel = str(Path(card_img_path).relative_to(UPLOAD_DIR))
                    card_data = {
                        "position": idx,
                        "row": row,
                        "col": col,
                        "card_id": result["card_id"],
                        "confidence": result["confidence"],
                        "method": result["method"],
                        "card_name": None,
                        "market_price": None,
                        "set_name": None,
                        "image_url": None,
                        "local_image_url": _local_image_url(result["card_id"]),
                        "segment_image_url": f"/segment-image/{seg_rel}",
                    }

                    if result["card_id"]:
                        row = session.execute(
                            sql_text("""
                                SELECT c.name, s.name as set_name, c.image_small, p.market_price
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
                            card_data["card_name"] = row.name
                            card_data["set_name"] = row.set_name
                            card_data["market_price"] = (
                                float(row.market_price) if row.market_price else None
                            )
                            card_data["image_url"] = row.image_small

                    cards.append(card_data)

        except Exception as e:
            logger.error("Page scan identification error: %s", e)
            self._send_json({"error": str(e), "cards": []}, status=500)
            return

        total_value = sum(c["market_price"] for c in cards if c["market_price"])
        self._send_json({
            "status": "ok",
            "cards": cards,
            "total_cards": len(cards),
            "total_value": round(total_value, 2),
        })

    def _send_price_history(self, card_id):
        """Return last 30 days of market prices for a card as JSON array."""
        try:
            from cardprice.db.session import SessionLocal
            from sqlalchemy import text as sql_text

            with SessionLocal() as session:
                rows = session.execute(
                    sql_text("""
                        SELECT price_date, market_price
                        FROM fact_market_prices
                        WHERE card_id = :cid
                        ORDER BY price_date DESC
                        LIMIT 30
                    """),
                    {"cid": card_id},
                ).fetchall()
                result = [
                    {
                        "date": str(r.price_date),
                        "price": float(r.market_price) if r.market_price else 0,
                    }
                    for r in rows
                ]
                self._send_json(result)
        except Exception as e:
            logger.error("Price history error: %s", e)
            self._send_json([])

    def _handle_inventory_add(self):
        """Add a card to user inventory (upsert).

        Accepts JSON: {"card_id": "base1-4/holofoil", "quantity": 1}
        Validates card exists in dim_cards, then upserts into user_inventory.
        """
        data = self._read_json_body()
        if data is None:
            return

        card_id = data.get("card_id")
        if not card_id:
            self._send_json({"error": "Missing required field: card_id"}, status=400)
            return

        quantity = data.get("quantity", 1)
        try:
            quantity = int(quantity)
        except (ValueError, TypeError):
            self._send_json({"error": "quantity must be an integer"}, status=400)
            return
        if quantity < 1:
            self._send_json({"error": "quantity must be >= 1"}, status=400)
            return

        try:
            from cardprice.db.session import SessionLocal
            from sqlalchemy import text as sql_text

            with SessionLocal() as session:
                # Validate card exists
                card = session.execute(
                    sql_text("SELECT card_id FROM dim_cards WHERE card_id = :cid"),
                    {"cid": card_id},
                ).fetchone()
                if not card:
                    self._send_json({"error": f"Card not found: {card_id}"}, status=404)
                    return

                # Upsert: increment if exists, insert otherwise
                existing = session.execute(
                    sql_text("""
                        SELECT id, quantity FROM user_inventory
                        WHERE card_id = :cid
                        ORDER BY id LIMIT 1
                    """),
                    {"cid": card_id},
                ).fetchone()

                if existing:
                    new_qty = existing.quantity + quantity
                    session.execute(
                        sql_text("""
                            UPDATE user_inventory
                            SET quantity = :qty, updated_at = NOW()
                            WHERE id = :rid
                        """),
                        {"qty": new_qty, "rid": existing.id},
                    )
                else:
                    new_qty = quantity
                    session.execute(
                        sql_text("""
                            INSERT INTO user_inventory (card_id, quantity, created_at, updated_at)
                            VALUES (:cid, :qty, NOW(), NOW())
                        """),
                        {"cid": card_id, "qty": new_qty},
                    )
                session.commit()

                self._send_json({
                    "card_id": card_id,
                    "quantity": new_qty,
                    "action": "added",
                })
        except Exception as e:
            logger.error("Inventory add error: %s", e)
            self._send_json({"error": str(e)}, status=500)

    def _handle_inventory_remove(self):
        """Remove a card from user inventory (decrement or delete).

        Accepts JSON: {"card_id": "base1-4/holofoil", "quantity": 1}
        Decrements quantity; deletes row if result <= 0.
        """
        data = self._read_json_body()
        if data is None:
            return

        card_id = data.get("card_id")
        if not card_id:
            self._send_json({"error": "Missing required field: card_id"}, status=400)
            return

        quantity = data.get("quantity", 1)
        try:
            quantity = int(quantity)
        except (ValueError, TypeError):
            self._send_json({"error": "quantity must be an integer"}, status=400)
            return
        if quantity < 1:
            self._send_json({"error": "quantity must be >= 1"}, status=400)
            return

        try:
            from cardprice.db.session import SessionLocal
            from sqlalchemy import text as sql_text

            with SessionLocal() as session:
                row = session.execute(
                    sql_text("SELECT quantity FROM user_inventory WHERE card_id = :cid"),
                    {"cid": card_id},
                ).fetchone()
                if not row:
                    self._send_json({"error": f"Card not in inventory: {card_id}"}, status=404)
                    return

                new_qty = row.quantity - quantity
                if new_qty <= 0:
                    session.execute(
                        sql_text("DELETE FROM user_inventory WHERE card_id = :cid"),
                        {"cid": card_id},
                    )
                    session.commit()
                    self._send_json({
                        "card_id": card_id,
                        "quantity": 0,
                        "action": "removed",
                    })
                else:
                    session.execute(
                        sql_text("""
                            UPDATE user_inventory
                            SET quantity = :qty, updated_at = NOW()
                            WHERE card_id = :cid
                        """),
                        {"cid": card_id, "qty": new_qty},
                    )
                    session.commit()
                    self._send_json({
                        "card_id": card_id,
                        "quantity": new_qty,
                        "action": "decremented",
                    })
        except Exception as e:
            logger.error("Inventory remove error: %s", e)
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

    def _send_csv_export(self):
        """Export inventory as CSV with columns: card_id, name, set_name, variant, quantity, condition, market_price, total_value."""
        try:
            from cardprice.db.session import SessionLocal
            from sqlalchemy import text as sql_text

            with SessionLocal() as session:
                rows = session.execute(sql_text("""
                    SELECT ui.card_id, dc.name, ds.name as set_name, ui.quantity,
                           ui.condition, lp.market_price
                    FROM user_inventory ui
                    JOIN dim_cards dc ON dc.card_id = ui.card_id
                    JOIN dim_sets ds ON ds.set_id = dc.set_id
                    LEFT JOIN LATERAL (
                        SELECT market_price FROM fact_market_prices
                        WHERE card_id = ui.card_id
                        ORDER BY price_date DESC LIMIT 1
                    ) lp ON true
                    ORDER BY ds.name, dc.name
                """)).fetchall()

                # Extract variant from card_id (format: setnum-cardnum/variant)
                csv_buffer = io.StringIO()
                writer = csv.writer(csv_buffer)

                # Write header
                writer.writerow([
                    "card_id",
                    "name",
                    "set_name",
                    "variant",
                    "quantity",
                    "condition",
                    "market_price",
                    "total_value"
                ])

                # Write data rows
                for r in rows:
                    card_id = r.card_id or ""
                    variant = ""
                    if "/" in card_id:
                        variant = card_id.split("/", 1)[1]

                    market_price = float(r.market_price) if r.market_price else 0.0
                    total_value = market_price * r.quantity

                    writer.writerow([
                        card_id,
                        r.name or "",
                        r.set_name or "",
                        variant,
                        r.quantity or 0,
                        r.condition or "",
                        f"{market_price:.2f}" if market_price > 0 else "",
                        f"{total_value:.2f}" if total_value > 0 else ""
                    ])

                csv_content = csv_buffer.getvalue()
                csv_bytes = csv_content.encode("utf-8")

                # Send CSV response with attachment headers
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", 'attachment; filename="inventory.csv"')
                self.send_header("Content-Length", str(len(csv_bytes)))
                self.end_headers()
                self.wfile.write(csv_bytes)

        except Exception as e:
            logger.error("CSV export error: %s", e)
            self._send_json({"error": str(e)}, status=500)

    def _send_pending(self):
        """List all pending scans awaiting identification."""
        pending = []
        for f in sorted(PENDING_DIR.glob("*.json")):
            data = json.loads(f.read_text())
            if data.get("status") == "pending":
                pending.append(data)
        self._send_json({"pending": pending, "count": len(pending)})

    def _send_history(self):
        """Return the last 50 scans (resolved and pending) sorted by timestamp desc."""
        scans = []
        for f in PENDING_DIR.glob("*.json"):
            try:
                data = json.loads(f.read_text())
            except (json.JSONDecodeError, OSError):
                continue

            scan_id = data.get("scan_id", f.stem)
            status = data.get("status", "unknown")

            # Derive timestamp from scan_id (format: YYYYMMDD_HHMMSS or page_YYYYMMDD_HHMMSS)
            ts_str = scan_id.replace("page_", "")
            try:
                ts = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
                timestamp = ts.isoformat()
            except ValueError:
                # Fallback: use file mtime
                timestamp = datetime.fromtimestamp(
                    os.path.getmtime(str(f))
                ).isoformat()

            entry = {
                "scan_id": scan_id,
                "status": status,
                "card_id": data.get("card_id"),
                "card_name": data.get("card_name"),
                "market_price": data.get("market_price"),
                "method": data.get("method"),
                "timestamp": timestamp,
            }
            scans.append(entry)

        # Sort by timestamp descending, take last 50
        scans.sort(key=lambda s: s["timestamp"], reverse=True)
        scans = scans[:50]

        # Enrich resolved scans that have a card_id but no price data from DB
        resolved_ids = [
            s["card_id"]
            for s in scans
            if s["status"] == "resolved" and s["card_id"] and s["market_price"] is None
        ]
        if resolved_ids:
            try:
                from cardprice.db.session import SessionLocal
                from sqlalchemy import text as sql_text

                with SessionLocal() as session:
                    rows = session.execute(
                        sql_text("""
                            SELECT c.card_id, c.name, p.market_price
                            FROM dim_cards c
                            LEFT JOIN LATERAL (
                                SELECT market_price FROM fact_market_prices
                                WHERE card_id = c.card_id
                                ORDER BY price_date DESC LIMIT 1
                            ) p ON true
                            WHERE c.card_id = ANY(:ids)
                        """),
                        {"ids": resolved_ids},
                    ).fetchall()
                    db_lookup = {
                        r.card_id: {"name": r.name, "price": float(r.market_price) if r.market_price else None}
                        for r in rows
                    }
                    for s in scans:
                        if s["card_id"] in db_lookup:
                            info = db_lookup[s["card_id"]]
                            if s["card_name"] is None:
                                s["card_name"] = info["name"]
                            if s["market_price"] is None:
                                s["market_price"] = info["price"]
            except Exception as e:
                logger.debug("History DB enrichment skipped: %s", e)

        self._send_json({"scans": scans, "count": len(scans)})

    def _send_stats(self):
        """Return scanning statistics computed from pending_scans JSON files."""
        scans = []
        for f in PENDING_DIR.glob("*.json"):
            try:
                scans.append(json.loads(f.read_text()))
            except (json.JSONDecodeError, OSError):
                continue

        total = len(scans)
        resolved = sum(1 for s in scans if s.get("status") == "resolved")
        pending = sum(1 for s in scans if s.get("status") == "pending")

        # Method breakdown and average confidence
        method_counts = {}
        method_conf_sums = {}
        method_conf_counts = {}
        for s in scans:
            method = s.get("method")
            if method:
                method_counts[method] = method_counts.get(method, 0) + 1
                conf = s.get("confidence")
                if conf is not None:
                    method_conf_sums[method] = method_conf_sums.get(method, 0.0) + conf
                    method_conf_counts[method] = method_conf_counts.get(method, 0) + 1

        avg_confidence_by_method = {}
        for m, total_conf in method_conf_sums.items():
            count = method_conf_counts[m]
            avg_confidence_by_method[m] = round(total_conf / count, 4)

        # ML index file sizes
        data_dir = Path("data")
        index_files = {
            "hash_db": data_dir / "hash_db.pkl",
            "dino_index": data_dir / "dino_index.faiss",
            "dino_card_ids": data_dir / "dino_card_ids.pkl",
            "clip_text_index": data_dir / "clip_text_index.pkl",
        }
        index_sizes = {}
        for name, path in index_files.items():
            if path.exists():
                size_bytes = path.stat().st_size
                index_sizes[name] = {
                    "bytes": size_bytes,
                    "human": (
                        f"{size_bytes / 1048576:.1f} MB"
                        if size_bytes >= 1048576
                        else f"{size_bytes / 1024:.1f} KB"
                    ),
                }

        # Count card images
        card_images_dir = data_dir / "card_images"
        image_count = 0
        if card_images_dir.exists():
            for entry in card_images_dir.iterdir():
                if entry.is_dir():
                    # set subdirectories contain the actual images
                    image_count += sum(
                        1 for _ in entry.iterdir() if _.is_file()
                    )
                elif entry.is_file():
                    image_count += 1

        self._send_json({
            "total_scans": total,
            "resolved": resolved,
            "pending": pending,
            "method_counts": method_counts,
            "avg_confidence_by_method": avg_confidence_by_method,
            "index_sizes": index_sizes,
            "card_image_count": image_count,
        })

    def _send_result(self, scan_id):
        """Check result of a specific scan by scan_id."""
        meta_path = PENDING_DIR / f"{scan_id}.json"
        if not meta_path.exists():
            self.send_error(404, "Scan not found")
            return
        data = json.loads(meta_path.read_text())
        # If resolved, look up price
        if data.get("status") == "resolved" and data.get("card_id"):
            try:
                from cardprice.db.session import SessionLocal
                from sqlalchemy import text as sql_text
                with SessionLocal() as session:
                    row = session.execute(
                        sql_text("""
                            SELECT c.name, s.name as set_name, c.image_small, p.market_price
                            FROM dim_cards c
                            JOIN dim_sets s ON s.set_id = c.set_id
                            LEFT JOIN LATERAL (
                                SELECT market_price FROM fact_market_prices
                                WHERE card_id = c.card_id
                                ORDER BY price_date DESC LIMIT 1
                            ) p ON true
                            WHERE c.card_id = :cid
                        """),
                        {"cid": data["card_id"]},
                    ).fetchone()
                    if row:
                        data["card_name"] = row.name
                        data["set_name"] = row.set_name
                        data["market_price"] = float(row.market_price) if row.market_price else None
                        data["image_url"] = row.image_small
                        data["local_image_url"] = _local_image_url(data["card_id"])
            except Exception as e:
                logger.error("Result lookup error: %s", e)
        self._send_json(data)

    def _stream_sse(self, scan_id):
        """Stream Server-Sent Events for a pending scan until it resolves or times out."""
        meta_path = PENDING_DIR / f"{scan_id}.json"
        if not meta_path.exists():
            self.send_error(404, "Scan not found")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        timeout = 5 * 60  # 5 minutes
        start = time.monotonic()
        last_keepalive = start

        try:
            while time.monotonic() - start < timeout:
                # Check current status
                try:
                    data = json.loads(meta_path.read_text())
                except (json.JSONDecodeError, OSError):
                    time.sleep(1)
                    continue

                if data.get("status") == "resolved":
                    # Enrich with DB info if needed
                    if data.get("card_id") and not data.get("card_name"):
                        try:
                            from cardprice.db.session import SessionLocal
                            from sqlalchemy import text as sql_text
                            with SessionLocal() as session:
                                row = session.execute(
                                    sql_text("""
                                        SELECT c.name, s.name as set_name, c.image_small, p.market_price
                                        FROM dim_cards c
                                        JOIN dim_sets s ON s.set_id = c.set_id
                                        LEFT JOIN LATERAL (
                                            SELECT market_price FROM fact_market_prices
                                            WHERE card_id = c.card_id
                                            ORDER BY price_date DESC LIMIT 1
                                        ) p ON true
                                        WHERE c.card_id = :cid
                                    """),
                                    {"cid": data["card_id"]},
                                ).fetchone()
                                if row:
                                    data["card_name"] = row.name
                                    data["set_name"] = row.set_name
                                    data["market_price"] = float(row.market_price) if row.market_price else None
                                    data["image_url"] = row.image_small
                                    data["local_image_url"] = _local_image_url(data["card_id"])
                        except Exception as e:
                            logger.debug("SSE DB enrichment error: %s", e)

                    payload = json.dumps(data)
                    self.wfile.write(f"event: resolved\ndata: {payload}\n\n".encode())
                    self.wfile.flush()
                    return

                # Send keepalive comment every 15 seconds
                now = time.monotonic()
                if now - last_keepalive >= 15:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    last_keepalive = now

                time.sleep(1)

            # Timeout reached
            self.wfile.write(b"event: timeout\ndata: {}\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            logger.debug("SSE client disconnected for scan %s", scan_id)

    def _send_card_image(self, card_path):
        """Serve a local card reference image as PNG.

        URL format: /card-image/bw5-107/normal
        Maps to:    data/card_images/bw5/bw5-107_normal.png

        The set_id is derived by stripping the trailing -<number> suffix from
        the card identifier (everything before the slash).
        """
        card_path = card_path.strip("/")
        if "/" not in card_path:
            self.send_error(400, "Expected format: <card_id>/<variant>")
            return

        base_id, variant = card_path.rsplit("/", 1)

        # Derive set_id: everything before the last '-'
        # e.g. "bw5-107" -> "bw5", "ecard3-H32" -> "ecard3", "swsh12pt5-160" -> "swsh12pt5"
        last_dash = base_id.rfind("-")
        if last_dash <= 0:
            self.send_error(400, "Cannot parse set from card_id")
            return
        set_id = base_id[:last_dash]

        # Build file path: data/card_images/<set_id>/<base_id>_<variant>.png
        filename = f"{base_id}_{variant}.png"
        image_path = CARD_IMAGES_DIR / set_id / filename

        if not image_path.is_file():
            self.send_error(404, f"Image not found: {set_id}/{filename}")
            return

        png_data = image_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(png_data)))
        self.send_header("Cache-Control", "public, max-age=86400")
        self.end_headers()
        self.wfile.write(png_data)

    def _send_segment_image(self, rel_path):
        """Serve a segmented card image from data/inbox/.

        URL format: /segment-image/page_20260228_120000_cards/card_00.png
        """
        rel_path = rel_path.strip("/")
        # Security: prevent directory traversal
        if ".." in rel_path or rel_path.startswith("/"):
            self.send_error(400, "Invalid path")
            return

        image_path = UPLOAD_DIR / rel_path
        if not image_path.is_file():
            self.send_error(404, f"Segment image not found: {rel_path}")
            return

        img_data = image_path.read_bytes()
        ext = image_path.suffix.lower()
        ctype = "image/png" if ext == ".png" else "image/jpeg"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(img_data)))
        self.send_header("Cache-Control", "public, max-age=3600")
        self.end_headers()
        self.wfile.write(img_data)

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
