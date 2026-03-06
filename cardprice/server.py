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
    GET  /condition  -> Condition assessment capture UI (4-angle wizard)
    GET  /condition/capture/<card_id> -> Per-card capture UI with card identity pre-filled
    POST /condition/photo/<card_id>/<step> -> Upload one photo, get immediate quality feedback
    GET  /condition/report/<card_id> -> Get combined condition report for a card
    POST /condition/assess -> Receive 4 photos, run condition assessment pipeline
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
<a href="/condition" class="upload-btn" style="display:block;background:#e94560;color:#fff;margin-top:20px;text-decoration:none;text-align:center;">Grade Card Condition</a>
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


def _parse_multipart_named(body, content_type):
    """Extract all named files from multipart/form-data body.

    Returns dict of {field_name: (filename, file_bytes)}.
    """
    m = re.search(r'boundary="?([^\s";]+)"?', content_type)
    if not m:
        return {}
    boundary = m.group(1).encode()
    parts = body.split(b"--" + boundary)
    result = {}
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
        name_match = re.search(r'name="([^"]*)"', header_block)
        fn_match = re.search(r'filename="([^"]*)"', header_block)
        if name_match and fn_match and fn_match.group(1):
            result[name_match.group(1)] = (fn_match.group(1), file_data)
        elif name_match and (not fn_match or not fn_match.group(1)):
            # Plain text field (no filename) — store as (None, raw_bytes)
            result[name_match.group(1)] = (None, file_data)
    return result


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


def _ref_image_path(card_id):
    """Return the Path to the local reference image for a card_id, or None.

    Looks for the normal-variant PNG in data/card_images/<set_id>/.
    """
    if not card_id:
        return None
    base_id = card_id.split("/")[0] if "/" in card_id else card_id
    last_dash = base_id.rfind("-")
    if last_dash <= 0:
        return None
    set_id = base_id[:last_dash]
    image_path = CARD_IMAGES_DIR / set_id / f"{base_id}_normal.png"
    return image_path if image_path.is_file() else None


def _corner_condition(ratio):
    """Map a corner whitening proxy ratio to a TCG condition abbreviation.

    Uses the same thresholds as edge whitening but slightly more lenient
    since corner wear is derived indirectly from edge measurements.
    """
    if ratio <= 0.0:
        return "NM", "Gem Mint"
    if ratio < 0.008:
        return "NM", "Near Mint"
    if ratio < 0.03:
        return "LP", "Lightly Played"
    if ratio < 0.07:
        return "MP", "Moderately Played"
    return "HP", "Heavily Played"


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
        elif self.path == "/condition":
            from cardprice.condition_ui import CONDITION_HTML
            self._send_html(CONDITION_HTML)
        elif self.path.startswith("/condition/capture/"):
            from urllib.parse import unquote
            card_id = unquote(self.path.split("/condition/capture/", 1)[1])
            self._send_condition_capture(card_id)
        elif self.path.startswith("/condition/report/"):
            from urllib.parse import unquote
            card_id = unquote(self.path.split("/condition/report/", 1)[1])
            self._send_condition_report(card_id)
        elif self.path.startswith("/condition/heatmap/"):
            self._send_condition_heatmap(self.path.split("/condition/heatmap/", 1)[1])
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
        elif self.path == "/condition/assess":
            self._handle_condition_assess()
        elif self.path.startswith("/condition/photo/"):
            self._handle_condition_photo()
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

    def _send_condition_heatmap(self, tmpdir_name):
        """Serve the defect heatmap PNG from a condition assessment temp dir.

        URL: /condition/heatmap/<tmpdir_name>
        The heatmap is rendered during /condition/assess and saved as heatmap.png.
        """
        import tempfile as _tf

        tmpdir_name = tmpdir_name.strip("/")
        # Security: only allow simple directory names (no traversal)
        if ".." in tmpdir_name or "/" in tmpdir_name or not tmpdir_name.startswith("condition_"):
            self.send_error(400, "Invalid heatmap path")
            return

        heatmap_path = Path(_tf.gettempdir()) / tmpdir_name / "heatmap.png"
        if not heatmap_path.is_file():
            self.send_error(404, "Heatmap not found (run /condition/assess first)")
            return

        png_data = heatmap_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(png_data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(png_data)

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

                        # Condition-adjusted prices for all raw conditions
                        if row.market_price:
                            from cardprice.models.condition_pricing import (
                                CONDITION_MULTIPLIERS_WITH_CI,
                            )
                            nm = float(row.market_price)
                            cond_prices = {}
                            for cond in ("NM", "LP", "MP", "HP", "DMG"):
                                mult, ci_lo, ci_hi = CONDITION_MULTIPLIERS_WITH_CI[cond]
                                cond_prices[cond] = {
                                    "price": round(nm * mult, 2),
                                    "multiplier": mult,
                                    "range_low": round(nm * ci_lo, 2),
                                    "range_high": round(nm * ci_hi, 2),
                                }
                            response["condition_prices"] = cond_prices
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

                        # Condition-adjusted prices for all raw conditions
                        if row.market_price:
                            from cardprice.models.condition_pricing import (
                                CONDITION_MULTIPLIERS_WITH_CI,
                            )
                            nm = float(row.market_price)
                            cond_prices = {}
                            for cond in ("NM", "LP", "MP", "HP", "DMG"):
                                mult, ci_lo, ci_hi = CONDITION_MULTIPLIERS_WITH_CI[cond]
                                cond_prices[cond] = {
                                    "price": round(nm * mult, 2),
                                    "multiplier": mult,
                                    "range_low": round(nm * ci_lo, 2),
                                    "range_high": round(nm * ci_hi, 2),
                                }
                            response["condition_prices"] = cond_prices
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

    def _handle_condition_assess(self):
        """Receive up to 4 card photos, run condition assessment pipeline.

        Accepts multipart/form-data with fields:
          - front (required): JPEG image of card front
          - back, oblique, edge (optional): additional angle images
          - card_id (optional): text field with known card_id to skip identification
            and enable surface defect comparison against the reference image

        Returns JSON with overall grade, sub-grades, defect annotations,
        and condition-adjusted pricing.
        """
        import tempfile

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
        # Condition assess can receive 4 images — allow up to 80 MB
        max_bytes = 80 * 1024 * 1024
        if length > max_bytes:
            self.send_error(413, f"Upload too large (max {max_bytes // (1024*1024)} MB)")
            return

        body = self.rfile.read(length)
        files = _parse_multipart_named(body, content_type)

        if "front" not in files:
            self._send_json(
                {"error": "At least a front image is required"}, status=400
            )
            return

        # Extract optional card_id text field
        supplied_card_id = None
        if "card_id" in files:
            _, raw = files["card_id"]
            supplied_card_id = raw.decode("utf-8", errors="replace").strip() or None

        # Save images to a temp directory
        tmpdir = tempfile.mkdtemp(prefix="condition_")
        saved_paths = {}
        angle_names = ["front", "back", "oblique", "edge"]
        for name in angle_names:
            if name not in files:
                continue
            filename, file_data = files[name]
            if filename is None:
                continue  # skip text fields
            ext = Path(filename).suffix or ".jpg"
            save_path = Path(tmpdir) / f"{name}{ext}"
            save_path.write_bytes(file_data)
            saved_paths[name] = str(save_path)

        logger.info(
            "Condition assess: received %d images (card_id=%s), saved to %s",
            len(saved_paths), supplied_card_id, tmpdir,
        )

        # --- Run assessment pipeline on the front image ---
        front_path = saved_paths.get("front")
        tmpdir_name = Path(tmpdir).name
        response = {
            "overall_grade": 0.0,
            "condition": "NM",
            "sub_grades": {
                "centering": 0.0,
                "surface": 0.0,
                "edges": 0.0,
                "corners": 0.0,
            },
            "defects": [],
            "card_id": supplied_card_id,
            "card_name": None,
            "set_name": None,
            "image_url": None,
            "local_image_url": None,
            "nm_price": None,
            "assessed_price": None,
            "angles_received": list(saved_paths.keys()),
            "temp_dir": tmpdir,
            "heatmap_url": None,
        }

        # Centering detector (HSV-based border measurement)
        try:
            from cardprice.ml.centering_detector import measure_centering
            centering = measure_centering(image_path=front_path)
            centering_score = centering.get("centering_score", 0.0)
            response["sub_grades"]["centering"] = round(centering_score, 1)
            response["centering_detail"] = {
                "lr": centering.get("front_lr", ""),
                "tb": centering.get("front_tb", ""),
                "confidence": centering.get("confidence", 0.0),
            }
        except Exception as e:
            logger.warning("Centering detector failed: %s", e)
            response["centering_detail"] = {"error": str(e)}

        # Edge whitening detector (LAB+HSV border wear)
        try:
            from cardprice.ml.edge_whitening import measure_edge_whitening
            whitening = measure_edge_whitening(front_path)
            tcg_cond = whitening.get("tcg_condition", "NM")
            # Map whitening result to an edge sub-grade (10 = NM, 7 = LP, 4 = MP, 2 = HP)
            edge_grade_map = {"NM": 9.5, "LP": 7.0, "MP": 4.5, "HP": 2.0}
            response["sub_grades"]["edges"] = edge_grade_map.get(tcg_cond, 5.0)
            response["whitening_detail"] = {
                "overall_ratio": whitening.get("overall_ratio", 0.0),
                "worst_edge": whitening.get("worst_edge", ""),
                "worst_ratio": whitening.get("worst_ratio", 0.0),
                "condition_label": whitening.get("condition_label", ""),
                "tcg_condition": tcg_cond,
            }

            # Corner wear: derive from the 4 corner regions of the edge data
            # Use the average of the two worst per-edge whitening ratios as a
            # proxy for corner condition (corners sit at edge intersections).
            try:
                edge_ratios = sorted(
                    [whitening["edges"][s]["whitening_ratio"]
                     for s in ("top", "bottom", "left", "right")],
                    reverse=True,
                )
                # Two worst edges contribute most to corner wear
                corner_ratio = (edge_ratios[0] + edge_ratios[1]) / 2
                corner_label, _ = _corner_condition(corner_ratio)
                corner_grade_map = {
                    "NM": 9.5, "LP": 7.0, "MP": 4.5, "HP": 2.0, "DMG": 1.0,
                }
                response["sub_grades"]["corners"] = corner_grade_map.get(
                    corner_label, 5.0
                )
                response["corner_detail"] = {
                    "proxy_ratio": round(corner_ratio, 6),
                    "condition": corner_label,
                }
            except Exception as ce:
                logger.warning("Corner grade derivation failed: %s", ce)
        except Exception as e:
            logger.warning("Edge whitening detector failed: %s", e)
            response["whitening_detail"] = {"error": str(e)}

        # --- Card identification (use supplied card_id or auto-detect) ---
        card_id = supplied_card_id
        try:
            from cardprice.db.session import SessionLocal
            from cardprice.models.condition_pricing import get_conditioned_price
            from sqlalchemy import text as sql_text

            if not card_id:
                # Auto-identify from the front image
                from cardprice.ml import identify_card
                with SessionLocal() as session:
                    id_result = identify_card(front_path, session=session)
                    card_id = id_result.get("card_id")
                    if card_id:
                        response["card_id"] = card_id
                        response["identification_confidence"] = id_result.get("confidence")
                        response["identification_method"] = id_result.get("method")
        except Exception as e:
            logger.warning("Condition assess: card identification failed: %s", e)

        # --- Surface defect detection (DINOv2 patch comparison) ---
        # Requires a reference image, so we need a known card_id
        ref_image_path = None
        if card_id:
            ref_image_path = _ref_image_path(card_id)

        if ref_image_path and ref_image_path.is_file():
            try:
                from cardprice.ml.surface_detector import (
                    detect_surface_defects,
                    estimate_condition,
                    render_heatmap,
                )

                surface_result = detect_surface_defects(
                    front_path, str(ref_image_path)
                )
                surface_cond = estimate_condition(surface_result)

                # Map surface grade: NM=9.5, LP=7, MP=4.5, HP=2, DMG=1
                surface_grade_map = {
                    "NM": 9.5, "LP": 7.0, "MP": 4.5, "HP": 2.0, "DMG": 1.0,
                }
                response["sub_grades"]["surface"] = surface_grade_map.get(
                    surface_cond["grade_abbrev"], 5.0
                )
                response["surface_detail"] = {
                    "defect_score": round(surface_result["defect_score"], 4),
                    "defect_count": surface_result["defect_count"],
                    "defect_ratio": round(surface_result["defect_ratio"], 4),
                    "mean_similarity": round(surface_result["mean_similarity"], 4),
                    "min_similarity": round(surface_result["min_similarity"], 4),
                    "grade": surface_cond["grade"],
                    "grade_abbrev": surface_cond["grade_abbrev"],
                    "confidence": surface_cond["confidence"],
                }

                # Serialize defect patch locations for the client
                response["defects"] = [
                    {"row": r, "col": c, "similarity": round(s, 4)}
                    for r, c, s in surface_result["defect_patches"][:20]  # top-20 worst
                ]

                # Render and save heatmap overlay for the visualization endpoint
                try:
                    heatmap_path = Path(tmpdir) / "heatmap.png"
                    render_heatmap(
                        surface_result["anomaly_map"],
                        output_path=str(heatmap_path),
                        title=f"Surface Defects — score={surface_result['defect_score']:.3f}",
                    )
                    if heatmap_path.is_file():
                        response["heatmap_url"] = f"/condition/heatmap/{tmpdir_name}"
                except Exception as he:
                    logger.warning("Heatmap render failed: %s", he)

            except Exception as e:
                logger.warning("Surface defect detector failed: %s", e)
                response["surface_detail"] = {"error": str(e)}
        else:
            msg = "no reference image" if card_id else "card not identified"
            response["surface_detail"] = {"skipped": msg}

        # --- Compute overall grade from available sub-grades ---
        # Only average sub-grades that have been populated (> 0)
        populated = [
            v for v in response["sub_grades"].values() if v > 0
        ]
        if populated:
            overall = sum(populated) / len(populated)
            response["overall_grade"] = round(overall, 1)
            # Map overall score to TCG condition
            if overall >= 8.5:
                response["condition"] = "NM"
            elif overall >= 6.5:
                response["condition"] = "LP"
            elif overall >= 4.0:
                response["condition"] = "MP"
            elif overall >= 2.0:
                response["condition"] = "HP"
            else:
                response["condition"] = "DMG"

        # --- Look up card metadata and apply condition-adjusted pricing ---
        if card_id:
            try:
                from cardprice.db.session import SessionLocal
                from cardprice.models.condition_pricing import get_conditioned_price
                from sqlalchemy import text as sql_text

                with SessionLocal() as session:
                    row = session.execute(
                        sql_text("""
                            SELECT c.name, s.name as set_name, c.image_small
                            FROM dim_cards c
                            JOIN dim_sets s ON s.set_id = c.set_id
                            WHERE c.card_id = :cid
                        """),
                        {"cid": card_id},
                    ).fetchone()
                    if row:
                        response["card_name"] = row.name
                        response["set_name"] = row.set_name
                        response["image_url"] = row.image_small
                        response["local_image_url"] = _local_image_url(card_id)

                    # Apply condition-adjusted pricing
                    pricing = get_conditioned_price(
                        card_id, response["condition"], session=session
                    )
                    response["nm_price"] = pricing["nm_price"]
                    response["assessed_price"] = pricing["assessed_price"]
                    response["multiplier"] = pricing["multiplier"]
                    response["price_range_low"] = pricing["price_range_low"]
                    response["price_range_high"] = pricing["price_range_high"]
                    response["price_date"] = pricing["price_date"]
            except Exception as e:
                logger.warning("Condition assess: pricing lookup failed: %s", e)

        self._send_json(response)

    def _send_condition_capture(self, card_id):
        """Serve the per-card 4-step condition capture UI.

        GET /condition/capture/<card_id>

        Looks up card metadata (name, set, image) and renders the capture
        wizard pre-filled with that card's identity.
        """
        card_name = None
        set_name = None
        image_url = None

        try:
            from cardprice.db.session import SessionLocal
            from sqlalchemy import text as sql_text

            with SessionLocal() as session:
                row = session.execute(
                    sql_text("""
                        SELECT c.name, s.name as set_name, c.image_small
                        FROM dim_cards c
                        JOIN dim_sets s ON s.set_id = c.set_id
                        WHERE c.card_id = :cid
                    """),
                    {"cid": card_id},
                ).fetchone()
                if row:
                    card_name = row.name
                    set_name = row.set_name
                    image_url = row.image_small
        except Exception as e:
            logger.warning("Condition capture: card lookup failed: %s", e)

        # Use local image URL if available
        local_url = _local_image_url(card_id)
        if local_url:
            image_url = local_url

        from cardprice.condition_ui import render_capture_html
        html = render_capture_html(card_id, card_name, set_name, image_url)
        self._send_html(html)

    def _handle_condition_photo(self):
        """Receive a single photo for one step, return immediate quality feedback.

        POST /condition/photo/<card_id>/<step>

        Accepts multipart/form-data with a single field 'photo'.
        Returns JSON with quality assessment:
          - quality: "good" | "acceptable" | "poor"
          - message: human-readable feedback
          - blur_score: Laplacian variance (higher = sharper)
          - brightness: mean pixel brightness (0-255)
        """
        # Parse card_id and step from path
        path_parts = self.path.split("/condition/photo/", 1)
        if len(path_parts) < 2:
            self._send_json({"error": "Invalid path"}, status=400)
            return

        from urllib.parse import unquote
        remainder = unquote(path_parts[1]).rstrip("/")
        # remainder is "<card_id>/<step>" where step is 0-3
        last_slash = remainder.rfind("/")
        if last_slash < 0:
            self._send_json({"error": "Missing step index in path"}, status=400)
            return

        card_id = remainder[:last_slash]
        step_str = remainder[last_slash + 1:]
        try:
            step_idx = int(step_str)
        except ValueError:
            self._send_json({"error": "Invalid step index"}, status=400)
            return

        step_names = ["front", "back", "oblique", "edge"]
        if step_idx < 0 or step_idx >= len(step_names):
            self._send_json({"error": "Step must be 0-3"}, status=400)
            return

        step_name = step_names[step_idx]

        # Read and parse the upload
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
        if length <= 0 or length > MAX_UPLOAD_BYTES:
            self.send_error(400, "Invalid upload size")
            return

        body = self.rfile.read(length)
        files = _parse_multipart_named(body, content_type)

        if "photo" not in files:
            self._send_json({"error": "No 'photo' field in upload"}, status=400)
            return

        filename, file_data = files["photo"]

        # Save to temp file for analysis
        import tempfile
        tmpdir = Path(tempfile.mkdtemp(prefix="condphoto_"))
        ext = Path(filename).suffix if filename else ".jpg"
        photo_path = tmpdir / f"{step_name}{ext}"
        photo_path.write_bytes(file_data)

        # Run quality checks
        quality = "good"
        message = ""
        blur_score = 0.0
        brightness = 128.0

        try:
            import cv2
            import numpy as np

            img = cv2.imread(str(photo_path))
            if img is None:
                self._send_json({
                    "quality": "poor",
                    "message": "Could not decode image",
                    "blur_score": 0,
                    "brightness": 0,
                    "step": step_name,
                })
                return

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Blur detection: Laplacian variance
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            blur_score = float(laplacian.var())

            # Brightness: mean of grayscale
            brightness = float(gray.mean())

            # Resolution check
            h, w = img.shape[:2]
            resolution_ok = min(h, w) >= 480

            # Quality thresholds
            issues = []

            if blur_score < 50:
                issues.append("Image is too blurry")
                quality = "poor"
            elif blur_score < 150:
                issues.append("Image is slightly soft")
                if quality == "good":
                    quality = "acceptable"

            if brightness < 40:
                issues.append("Image is too dark")
                quality = "poor"
            elif brightness < 70:
                issues.append("Image is a bit dark")
                if quality == "good":
                    quality = "acceptable"
            elif brightness > 230:
                issues.append("Image is overexposed")
                quality = "poor"
            elif brightness > 200:
                issues.append("Image is a bit bright")
                if quality == "good":
                    quality = "acceptable"

            if not resolution_ok:
                issues.append("Resolution is low")
                if quality == "good":
                    quality = "acceptable"

            # Step-specific checks
            if step_name == "oblique":
                # For oblique, we actually expect some glare/highlights
                # Check if there are bright spots (potential reflections)
                bright_pixels = np.sum(gray > 220) / gray.size
                if bright_pixels < 0.01 and quality == "good":
                    issues.append("No reflections visible - try angling toward light")
                    quality = "acceptable"

            if quality == "good":
                good_messages = {
                    "front": "Sharp and well-lit - good for centering analysis",
                    "back": "Clear back image - good for whitening detection",
                    "oblique": "Good angle capture - reflections visible for scratch detection",
                    "edge": "Clear edge view - good for corner and edge wear analysis",
                }
                message = good_messages.get(step_name, "Good quality capture")
            elif quality == "acceptable":
                message = ". ".join(issues) + " - usable but consider retaking"
            else:
                message = ". ".join(issues) + " - please retake"

        except ImportError:
            # cv2 not available -- skip quality checks, accept the photo
            quality = "good"
            message = "Photo received (quality check unavailable)"
        except Exception as e:
            logger.warning("Photo quality check failed: %s", e)
            quality = "acceptable"
            message = "Could not fully assess quality"

        # Clean up temp file
        try:
            photo_path.unlink(missing_ok=True)
            tmpdir.rmdir()
        except Exception:
            pass

        logger.info(
            "Condition photo: card_id=%s step=%s quality=%s blur=%.1f brightness=%.1f",
            card_id, step_name, quality, blur_score, brightness,
        )

        self._send_json({
            "quality": quality,
            "message": message,
            "blur_score": round(blur_score, 1),
            "brightness": round(brightness, 1),
            "step": step_name,
            "card_id": card_id,
        })

    def _send_condition_report(self, card_id):
        """Return the most recent condition assessment for a card.

        GET /condition/report/<card_id>

        Looks for cached condition results in the pending_scans directory
        and returns the most recent one matching the given card_id.
        If no cached report exists, returns a stub with the card metadata
        and instructions to run the capture workflow.
        """
        import tempfile as _tf

        # Search temp directories for the most recent condition assessment
        # that matches this card_id
        tmpdir_root = Path(_tf.gettempdir())
        best_report = None
        best_mtime = 0

        for d in tmpdir_root.glob("condition_*"):
            if not d.is_dir():
                continue
            front_path = d / "front.jpg"
            if not front_path.exists():
                # Also check for .jpeg extension
                front_path = d / "front.jpeg"
                if not front_path.exists():
                    continue

            # Check modification time
            mtime = front_path.stat().st_mtime
            if mtime > best_mtime:
                best_mtime = mtime
                best_report = d

        # Look up card metadata
        card_name = None
        set_name = None
        image_url = None
        nm_price = None

        try:
            from cardprice.db.session import SessionLocal
            from sqlalchemy import text as sql_text

            with SessionLocal() as session:
                row = session.execute(
                    sql_text("""
                        SELECT c.name, s.name as set_name, c.image_small
                        FROM dim_cards c
                        JOIN dim_sets s ON s.set_id = c.set_id
                        WHERE c.card_id = :cid
                    """),
                    {"cid": card_id},
                ).fetchone()
                if row:
                    card_name = row.name
                    set_name = row.set_name
                    image_url = row.image_small

                # Get latest NM price
                price_row = session.execute(
                    sql_text("""
                        SELECT market_price
                        FROM fact_prices
                        WHERE card_id = :cid
                        ORDER BY price_date DESC
                        LIMIT 1
                    """),
                    {"cid": card_id},
                ).fetchone()
                if price_row:
                    nm_price = float(price_row.market_price) if price_row.market_price else None
        except Exception as e:
            logger.warning("Condition report: card lookup failed: %s", e)

        local_url = _local_image_url(card_id)
        if local_url:
            image_url = local_url

        response = {
            "card_id": card_id,
            "card_name": card_name,
            "set_name": set_name,
            "image_url": image_url,
            "nm_price": nm_price,
            "has_report": False,
            "capture_url": f"/condition/capture/{card_id}",
        }

        # If we found a matching condition directory, try to re-derive the report
        # by running the assessor on the saved front image
        if best_report:
            front_path = best_report / "front.jpg"
            if not front_path.exists():
                front_path = best_report / "front.jpeg"

            if front_path.exists():
                try:
                    from cardprice.ml.condition_assessor import assess_condition
                    # Build images dict (assess_condition requires multi-photo)
                    images = {"front": str(front_path)}
                    for angle in ("back", "oblique", "edge"):
                        for ext in ("jpg", "jpeg", "png"):
                            p = best_report / f"{angle}.{ext}"
                            if p.exists():
                                images[angle] = str(p)
                                break
                    result = assess_condition(
                        images,
                        card_id=card_id,
                    )
                    overall_grade = result.get("overall_grade", "NM")
                    price_mult = result.get("price_multiplier", 1.0)

                    response["has_report"] = True
                    response["overall_grade"] = overall_grade
                    response["overall_confidence"] = result.get("overall_confidence", 0.0)
                    response["sub_scores"] = result.get("sub_scores", {})
                    response["modules_run"] = result.get("modules_run", [])
                    response["price_multiplier"] = price_mult

                    if nm_price is not None:
                        response["assessed_price"] = round(nm_price * price_mult, 2)
                    else:
                        response["assessed_price"] = None

                except Exception as e:
                    logger.warning("Condition report: assessment failed: %s", e)
                    response["error"] = f"Assessment failed: {e}"

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


def warmup():
    """Pre-load all ML models and data files so the first request is fast.

    Loads resources in dependency order:
      1. PaddleOCR (name OCR) + dummy inference to trigger JIT compilation
      2. EasyOCR (attack OCR fallback)
      3. DINOv2 + dummy inference to trigger JIT/CUDA warmup
      4. FAISS index + card_ids
      5. Reference embeddings
      6. Card names JSON
      7. Card attacks JSON

    Each step is timed and logged.  Failures are logged as warnings but do
    not prevent the server from starting.

    IMPORTANT: CLIP is deliberately NOT loaded here.  Loading CLIP alongside
    PaddlePaddle causes a SIGSEGV (segmentation fault) due to conflicting
    protobuf/ONNX internals.  CLIP is not used in the v2 pipeline anyway.
    """
    total_start = time.time()
    logger.info("=== ML warmup starting ===")

    # --- 1. PaddleOCR (name OCR) + dummy inference ---

    def _warmup_paddleocr():
        from cardprice.ml.ocr_matcher import _paddle_ocr_name
        import numpy as np
        # Trigger PaddleOCR model load AND first inference (JIT warmup)
        dummy = np.zeros((100, 300, 3), dtype=np.uint8)
        _paddle_ocr_name(dummy, 100, 300)

    # --- 2. EasyOCR (attack OCR fallback) ---

    def _warmup_easyocr_attack_ocr():
        """Load shared EasyOCR reader (used by attack_ocr + hp_detector)."""
        from cardprice.ml.ocr_matcher import get_easyocr_reader
        get_easyocr_reader()

    # --- 3. DINOv2 + dummy inference ---

    def _warmup_dinov2():
        from cardprice.ml.dino_matcher import _load_model, _transform
        import torch
        from PIL import Image
        model, device = _load_model()
        # Run a dummy inference to trigger CUDA/JIT warmup
        dummy_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        tensor = _transform(dummy_img).unsqueeze(0).to(device)
        with torch.no_grad():
            model(tensor)

    # --- 4. FAISS index + card_ids ---

    def _warmup_dino_faiss():
        from cardprice.ml import _get_dino_index
        _get_dino_index()

    # --- 5. Reference embeddings ---

    def _warmup_ref_embeddings():
        from cardprice.ml.ref_matcher import _load_ref_embeddings
        _load_ref_embeddings()

    # --- 6. Card names JSON ---

    def _warmup_card_names():
        from cardprice.ml.ocr_matcher import _load_card_names
        _load_card_names()

    # --- 7. Card attacks JSON ---

    def _warmup_attack_index():
        from cardprice.ml.attack_ocr import _load_attack_index
        _load_attack_index()

    # --- Supplementary (nice to have, not critical path) ---

    def _warmup_card_names_fallback():
        from cardprice.ml.ref_matcher import _load_card_names_fallback
        _load_card_names_fallback()

    def _warmup_card_metadata():
        from cardprice.ml import _get_card_metadata
        _get_card_metadata()

    def _warmup_hash_db():
        from cardprice.ml import _get_hash_db
        _get_hash_db()

    # Ordered: critical models first with dummy inference, then data files.
    # CLIP is deliberately excluded (SIGSEGV with PaddlePaddle).
    steps = [
        ("PaddleOCR (PP-OCRv5)",       _warmup_paddleocr),
        ("EasyOCR (attack_ocr)",       _warmup_easyocr_attack_ocr),
        ("DINOv2 ViT-B/14",           _warmup_dinov2),
        ("FAISS index (DINOv2)",       _warmup_dino_faiss),
        ("Ref embeddings (DINOv2)",    _warmup_ref_embeddings),
        ("Card names (DB/JSON)",       _warmup_card_names),
        ("Attack index",               _warmup_attack_index),
        ("Card names fallback (JSON)", _warmup_card_names_fallback),
        ("Card metadata (DB)",         _warmup_card_metadata),
        ("Hash DB",                    _warmup_hash_db),
    ]

    loaded = 0
    failed = 0
    for name, fn in steps:
        step_start = time.time()
        try:
            fn()
            elapsed = time.time() - step_start
            logger.info("  [OK] %-30s  %.1fs", name, elapsed)
            loaded += 1
        except Exception as e:
            elapsed = time.time() - step_start
            logger.warning("  [FAIL] %-30s  %.1fs — %s", name, elapsed, e)
            failed += 1

    total = time.time() - total_start
    logger.info("=== ML warmup complete: %d loaded, %d failed, %.1fs total ===",
                loaded, failed, total)


def run_server(host="0.0.0.0", port=8888):
    """Start the HTTP server."""
    global _server_port
    _server_port = port
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    warmup()

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
