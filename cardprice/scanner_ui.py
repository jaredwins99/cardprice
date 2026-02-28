"""Multi-card scanner UI for binder page and single card scanning.

Exports MULTI_CARD_HTML: a self-contained HTML page (no external deps)
that provides:
- Mode toggle: Single Card vs Binder Page
- Grid display of identified cards with reference images, prices
- Total page value summary
- Tap-to-detail modal for individual cards
- Pending/queued status with spinner animation
- Mobile-first dark theme matching the existing scanner style
"""

MULTI_CARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=no">
<title>Card Scanner</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --bg-primary: #1a1a2e;
    --bg-card: #16213e;
    --bg-card-hover: #1c2a4a;
    --bg-modal: #0f1629;
    --accent: #e94560;
    --accent-dark: #c23152;
    --green: #4ecca3;
    --green-dim: #3ba882;
    --text: #eee;
    --text-dim: #888;
    --text-faint: #555;
    --radius: 12px;
    --radius-sm: 8px;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg-primary);
    color: var(--text);
    min-height: 100vh;
    -webkit-tap-highlight-color: transparent;
}

.container {
    max-width: 600px;
    margin: 0 auto;
    padding: 16px 12px 100px;
}

/* Header */
.header {
    text-align: center;
    padding: 8px 0 16px;
}
.header h1 {
    color: var(--accent);
    font-size: 22px;
    font-weight: 700;
    letter-spacing: -0.5px;
}

/* QR Section */
.qr-section {
    text-align: center;
    background: var(--bg-card);
    border-radius: var(--radius);
    padding: 15px;
    margin: 0 0 16px;
}
.qr-section p { margin: 5px 0 10px; color: var(--text-dim); font-size: 14px; }
.qr-section .url { font-family: monospace; color: var(--green); font-size: 13px; word-break: break-all; }
#qrCanvas { image-rendering: pixelated; border-radius: 4px; }

/* Mode Toggle */
.mode-toggle {
    display: flex;
    background: var(--bg-card);
    border-radius: var(--radius);
    padding: 4px;
    margin-bottom: 16px;
    gap: 4px;
}
.mode-btn {
    flex: 1;
    padding: 12px 8px;
    border: none;
    border-radius: var(--radius-sm);
    background: transparent;
    color: var(--text-dim);
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
}
.mode-btn.active {
    background: var(--accent);
    color: #fff;
}

/* Upload Buttons */
.upload-area {
    display: flex;
    gap: 8px;
    margin-bottom: 16px;
}
.upload-btn {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 16px 12px;
    font-size: 15px;
    font-weight: 600;
    border: none;
    border-radius: var(--radius);
    cursor: pointer;
    transition: background 0.15s;
}
.upload-btn.primary {
    background: var(--accent);
    color: #fff;
}
.upload-btn.primary:active { background: var(--accent-dark); }
.upload-btn.secondary {
    background: var(--bg-card);
    color: var(--text);
    border: 2px solid var(--accent);
}
.upload-btn.secondary:active { background: var(--bg-card-hover); }
input[type=file] { display: none; }

/* Preview */
#preview {
    width: 100%;
    max-height: 300px;
    object-fit: contain;
    border-radius: var(--radius-sm);
    margin-bottom: 16px;
    display: none;
}

/* Summary Bar */
.summary-bar {
    display: none;
    background: var(--bg-card);
    border-radius: var(--radius);
    padding: 14px 16px;
    margin-bottom: 16px;
    justify-content: space-between;
    align-items: center;
}
.summary-bar.show { display: flex; }
.summary-label { color: var(--text-dim); font-size: 13px; }
.summary-value { font-size: 22px; font-weight: 700; color: var(--green); }
.summary-count { font-size: 14px; color: var(--text-dim); }

/* Scanning Indicator */
.scanning-indicator {
    display: none;
    text-align: center;
    padding: 24px;
    color: var(--text-dim);
    font-size: 15px;
}
.scanning-indicator.show { display: block; }
.scanning-indicator .dots::after {
    content: '';
    animation: dots 1.5s steps(4, end) infinite;
}
@keyframes dots {
    0%   { content: ''; }
    25%  { content: '.'; }
    50%  { content: '..'; }
    75%  { content: '...'; }
    100% { content: ''; }
}

/* Spinner */
@keyframes spin { to { transform: rotate(360deg); } }
.spinner-ring {
    display: inline-block;
    width: 20px; height: 20px;
    border: 2.5px solid var(--text-faint);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    vertical-align: middle;
    margin-right: 8px;
}

/* Card Grid */
.card-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
}
@media (max-width: 400px) {
    .card-grid { grid-template-columns: repeat(2, 1fr); }
}

/* Card Tile */
.card-tile {
    background: var(--bg-card);
    border-radius: var(--radius-sm);
    overflow: hidden;
    cursor: pointer;
    transition: transform 0.15s, box-shadow 0.15s;
    position: relative;
}
.card-tile:active {
    transform: scale(0.97);
}
.card-tile .tile-img-wrap {
    width: 100%;
    aspect-ratio: 3/4.2;
    background: #0d1321;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
}
.card-tile .tile-img-wrap img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}
.card-tile .tile-info {
    padding: 6px 8px 8px;
}
.card-tile .tile-name {
    font-size: 11px;
    font-weight: 600;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-bottom: 2px;
}
.card-tile .tile-set {
    font-size: 10px;
    color: var(--text-dim);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-bottom: 4px;
}
.card-tile .tile-price {
    font-size: 13px;
    font-weight: 700;
    color: var(--green);
}

/* Queued tile state */
.card-tile.queued .tile-img-wrap {
    position: relative;
}
.card-tile.queued .tile-img-wrap::after {
    content: '';
    position: absolute;
    width: 28px; height: 28px;
    border: 3px solid var(--text-faint);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
}
.card-tile.queued .tile-name { color: var(--text-dim); }

/* Single Card Result */
.single-result {
    display: none;
    background: var(--bg-card);
    border-radius: var(--radius);
    padding: 20px 16px;
    text-align: center;
}
.single-result.show { display: block; }
.single-result .sr-image {
    max-width: 200px;
    border-radius: var(--radius-sm);
    margin: 0 auto 16px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}
.single-result .sr-name {
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 4px;
}
.single-result .sr-set {
    font-size: 14px;
    color: var(--text-dim);
    margin-bottom: 8px;
}
.single-result .sr-price {
    font-size: 28px;
    font-weight: 700;
    color: var(--green);
    margin-bottom: 6px;
}
.single-result .sr-meta {
    font-size: 12px;
    color: var(--text-dim);
}

/* Detail Modal */
.modal-overlay {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.75);
    z-index: 1000;
    align-items: flex-end;
    justify-content: center;
}
.modal-overlay.show {
    display: flex;
}
.modal-sheet {
    background: var(--bg-modal);
    width: 100%;
    max-width: 600px;
    max-height: 85vh;
    border-radius: 20px 20px 0 0;
    padding: 8px 20px 32px;
    overflow-y: auto;
    animation: slideUp 0.25s ease-out;
}
@keyframes slideUp {
    from { transform: translateY(100%); }
    to { transform: translateY(0); }
}
.modal-handle {
    width: 40px;
    height: 4px;
    background: var(--text-faint);
    border-radius: 2px;
    margin: 8px auto 20px;
}
.modal-img {
    display: block;
    max-width: 220px;
    margin: 0 auto 16px;
    border-radius: var(--radius-sm);
    box-shadow: 0 4px 24px rgba(0,0,0,0.5);
}
.modal-name {
    font-size: 20px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 4px;
}
.modal-set {
    font-size: 14px;
    color: var(--text-dim);
    text-align: center;
    margin-bottom: 12px;
}
.modal-price {
    font-size: 32px;
    font-weight: 700;
    color: var(--green);
    text-align: center;
    margin-bottom: 6px;
}
.modal-meta-row {
    display: flex;
    justify-content: space-between;
    padding: 10px 0;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    font-size: 14px;
}
.modal-meta-row .label { color: var(--text-dim); }
.modal-meta-row .value { font-weight: 600; }
.modal-close {
    display: block;
    width: 100%;
    margin-top: 20px;
    padding: 14px;
    font-size: 16px;
    font-weight: 600;
    border: none;
    border-radius: var(--radius);
    background: var(--accent);
    color: #fff;
    cursor: pointer;
}
.modal-close:active { background: var(--accent-dark); }
</style>
</head>
<body>

<div class="container">
    <div class="header">
        <h1>Pokemon Card Scanner</h1>
    </div>

    <!-- QR Code (desktop only) -->
    <div class="qr-section" id="qrSection">
        <p>Scan QR code to open on your phone</p>
        <canvas id="qrCanvas"></canvas>
        <br>
        <span class="url" id="serverUrl"></span>
    </div>

    <!-- Mode Toggle -->
    <div class="mode-toggle">
        <button class="mode-btn active" data-mode="single" onclick="setMode('single')">Single Card</button>
        <button class="mode-btn" data-mode="binder" onclick="setMode('binder')">Binder Page</button>
    </div>

    <!-- Upload Buttons -->
    <div class="upload-area">
        <label class="upload-btn primary" for="camera">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M23 19a2 2 0 01-2 2H3a2 2 0 01-2-2V8a2 2 0 012-2h4l2-3h6l2 3h4a2 2 0 012 2z"/><circle cx="12" cy="13" r="4"/></svg>
            Take Photo
        </label>
        <input type="file" id="camera" accept="image/*" capture="environment">
        <label class="upload-btn secondary" for="gallery">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg>
            Gallery
        </label>
        <input type="file" id="gallery" accept="image/*">
    </div>

    <!-- Image Preview -->
    <img id="preview">

    <!-- Scanning Indicator -->
    <div class="scanning-indicator" id="scanningIndicator">
        <span class="spinner-ring"></span>
        <span>Scanning<span class="dots"></span></span>
    </div>

    <!-- Single Card Result -->
    <div class="single-result" id="singleResult">
        <img class="sr-image" id="srImage">
        <div class="sr-name" id="srName"></div>
        <div class="sr-set" id="srSet"></div>
        <div class="sr-price" id="srPrice"></div>
        <div class="sr-meta" id="srMeta"></div>
    </div>

    <!-- Binder Page Summary -->
    <div class="summary-bar" id="summaryBar">
        <div>
            <div class="summary-label">Page Total</div>
            <div class="summary-value" id="summaryTotal">$0.00</div>
        </div>
        <div style="text-align:right">
            <div class="summary-count" id="summaryCount">0 cards</div>
            <div class="summary-label" id="summaryStatus"></div>
        </div>
    </div>

    <!-- Card Grid (binder mode) -->
    <div class="card-grid" id="cardGrid"></div>
</div>

<!-- Detail Modal -->
<div class="modal-overlay" id="detailModal">
    <div class="modal-sheet" id="modalSheet">
        <div class="modal-handle"></div>
        <img class="modal-img" id="modalImg">
        <div class="modal-name" id="modalName"></div>
        <div class="modal-set" id="modalSet"></div>
        <div class="modal-price" id="modalPrice"></div>
        <div id="modalMeta">
            <div class="modal-meta-row">
                <span class="label">Card ID</span>
                <span class="value" id="modalCardId">--</span>
            </div>
            <div class="modal-meta-row">
                <span class="label">Method</span>
                <span class="value" id="modalMethod">--</span>
            </div>
            <div class="modal-meta-row">
                <span class="label">Confidence</span>
                <span class="value" id="modalConfidence">--</span>
            </div>
        </div>
        <button class="modal-close" onclick="closeModal()">Close</button>
    </div>
</div>

<!-- QR Code Generator (same as existing, self-contained) -->
<script>
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
    var sz=m.length;var bits=[1,0,1,0,1,0,0,0,0,0,1,0,0,1,0];
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
(function(){
    var url='http://'+location.host;
    document.getElementById('serverUrl').textContent=url;
    drawQR('qrCanvas',url,6);
    if(/Mobi|Android|iPhone|iPad/i.test(navigator.userAgent))
        document.getElementById('qrSection').style.display='none';
})();
</script>

<script>
(function() {
    "use strict";

    // ---- State ----
    var currentMode = 'single';  // 'single' or 'binder'
    var cards = [];               // Array of card result objects for binder mode
    var activePolls = {};         // scan_id -> intervalId

    // ---- Mode Toggle ----
    window.setMode = function(mode) {
        currentMode = mode;
        var btns = document.querySelectorAll('.mode-btn');
        for (var i = 0; i < btns.length; i++) {
            btns[i].classList.toggle('active', btns[i].getAttribute('data-mode') === mode);
        }
        // Reset display
        document.getElementById('singleResult').classList.remove('show');
        document.getElementById('summaryBar').classList.remove('show');
        document.getElementById('cardGrid').innerHTML = '';
        document.getElementById('preview').style.display = 'none';
        document.getElementById('scanningIndicator').classList.remove('show');
        cards = [];
        clearAllPolls();
    };

    function clearAllPolls() {
        for (var id in activePolls) {
            clearInterval(activePolls[id]);
        }
        activePolls = {};
    }

    // ---- File Handling ----
    function handleFile(file) {
        if (!file) return;
        var preview = document.getElementById('preview');
        preview.src = URL.createObjectURL(file);
        preview.style.display = 'block';

        if (currentMode === 'single') {
            handleSingleScan(file);
        } else {
            handleBinderScan(file);
        }
    }

    // ---- Single Card Mode ----
    function handleSingleScan(file) {
        var indicator = document.getElementById('scanningIndicator');
        var result = document.getElementById('singleResult');
        var summary = document.getElementById('summaryBar');
        var grid = document.getElementById('cardGrid');

        indicator.classList.add('show');
        result.classList.remove('show');
        summary.classList.remove('show');
        grid.innerHTML = '';

        var fd = new FormData();
        fd.append('image', file);

        fetch('/scan', { method: 'POST', body: fd })
            .then(function(r) { return r.json(); })
            .then(function(data) {
                indicator.classList.remove('show');
                if (data.status === 'pending') {
                    showSinglePending(data.scan_id);
                } else if (data.error) {
                    showSingleError(data.error);
                } else {
                    showSingleResult(data);
                }
            })
            .catch(function(e) {
                indicator.classList.remove('show');
                showSingleError(String(e));
            });
    }

    function showSingleResult(data) {
        var el = document.getElementById('singleResult');
        el.classList.add('show');

        var img = document.getElementById('srImage');
        if (data.image_url) {
            img.src = data.image_url;
            img.style.display = 'block';
        } else {
            img.style.display = 'none';
        }

        document.getElementById('srName').textContent = data.card_name || 'Unknown Card';
        document.getElementById('srSet').textContent = data.set_name || '';
        document.getElementById('srPrice').textContent = data.market_price ? '$' + Number(data.market_price).toFixed(2) : 'No price data';
        var metaParts = [];
        if (data.confidence) metaParts.push(Math.round(data.confidence * 100) + '% confidence');
        if (data.method) metaParts.push('via ' + data.method);
        if (data.card_id) metaParts.push(data.card_id);
        document.getElementById('srMeta').textContent = metaParts.join(' \u2022 ');
    }

    function showSinglePending(scanId) {
        var el = document.getElementById('singleResult');
        el.classList.add('show');
        document.getElementById('srImage').style.display = 'none';
        document.getElementById('srName').textContent = 'Queued for identification...';
        document.getElementById('srSet').textContent = '';
        document.getElementById('srPrice').innerHTML = '<span class="spinner-ring"></span> Checking';
        document.getElementById('srMeta').textContent = 'Polling every 3s';

        var poll = setInterval(function() {
            fetch('/result/' + scanId)
                .then(function(r) { return r.json(); })
                .then(function(data) {
                    if (data.status === 'resolved') {
                        clearInterval(poll);
                        showSingleResult(data);
                    }
                });
        }, 3000);
        activePolls[scanId] = poll;
    }

    function showSingleError(msg) {
        var el = document.getElementById('singleResult');
        el.classList.add('show');
        document.getElementById('srImage').style.display = 'none';
        document.getElementById('srName').textContent = 'Error';
        document.getElementById('srSet').textContent = msg;
        document.getElementById('srPrice').textContent = '';
        document.getElementById('srMeta').textContent = '';
    }

    // ---- Binder Page Mode ----
    // For binder mode, we send the same /scan endpoint.
    // The server currently handles one image = one card.
    // In binder mode, we post the image once, and the response may contain
    // a single card (the server could be extended to return multiple).
    // For forward-compatibility, we handle both array and single responses.
    // We also support the user scanning multiple photos in succession to
    // build up the binder page grid (e.g., one photo per card slot).

    function handleBinderScan(file) {
        var indicator = document.getElementById('scanningIndicator');
        indicator.classList.add('show');
        document.getElementById('singleResult').classList.remove('show');

        // Add a placeholder tile while scanning
        var placeholderIdx = cards.length;
        cards.push({ status: 'scanning' });
        renderGrid();

        var fd = new FormData();
        fd.append('image', file);

        fetch('/scan', { method: 'POST', body: fd })
            .then(function(r) { return r.json(); })
            .then(function(data) {
                indicator.classList.remove('show');

                // Handle array response (future multi-card endpoint)
                if (Array.isArray(data.cards)) {
                    // Remove placeholder
                    cards.splice(placeholderIdx, 1);
                    for (var i = 0; i < data.cards.length; i++) {
                        cards.push(normalizeCard(data.cards[i]));
                    }
                } else if (data.status === 'pending') {
                    cards[placeholderIdx] = {
                        status: 'pending',
                        scan_id: data.scan_id,
                        card_name: 'Queued...',
                    };
                    startPollForTile(placeholderIdx, data.scan_id);
                } else if (data.error) {
                    cards[placeholderIdx] = {
                        status: 'error',
                        card_name: 'Error',
                        set_name: data.error,
                    };
                } else {
                    cards[placeholderIdx] = normalizeCard(data);
                }
                renderGrid();
                updateSummary();
            })
            .catch(function(e) {
                indicator.classList.remove('show');
                cards[placeholderIdx] = {
                    status: 'error',
                    card_name: 'Error',
                    set_name: String(e),
                };
                renderGrid();
                updateSummary();
            });
    }

    function normalizeCard(data) {
        return {
            status: 'resolved',
            card_id: data.card_id || null,
            card_name: data.card_name || 'Unknown Card',
            set_name: data.set_name || '',
            market_price: data.market_price || null,
            image_url: data.image_url || null,
            confidence: data.confidence || null,
            method: data.method || null,
        };
    }

    function startPollForTile(idx, scanId) {
        var poll = setInterval(function() {
            fetch('/result/' + scanId)
                .then(function(r) { return r.json(); })
                .then(function(data) {
                    if (data.status === 'resolved') {
                        clearInterval(poll);
                        delete activePolls[scanId];
                        cards[idx] = normalizeCard(data);
                        renderGrid();
                        updateSummary();
                    }
                });
        }, 3000);
        activePolls[scanId] = poll;
    }

    // ---- Grid Rendering ----
    function renderGrid() {
        var grid = document.getElementById('cardGrid');
        grid.innerHTML = '';

        for (var i = 0; i < cards.length; i++) {
            var card = cards[i];
            var tile = document.createElement('div');
            tile.className = 'card-tile';
            if (card.status === 'scanning' || card.status === 'pending') {
                tile.className += ' queued';
            }

            var imgWrap = document.createElement('div');
            imgWrap.className = 'tile-img-wrap';

            if (card.image_url) {
                var img = document.createElement('img');
                img.src = card.image_url;
                img.alt = card.card_name || '';
                img.loading = 'lazy';
                imgWrap.appendChild(img);
            }
            tile.appendChild(imgWrap);

            var info = document.createElement('div');
            info.className = 'tile-info';

            var name = document.createElement('div');
            name.className = 'tile-name';
            if (card.status === 'scanning') {
                name.textContent = 'Scanning...';
            } else if (card.status === 'pending') {
                name.textContent = 'Queued...';
            } else {
                name.textContent = card.card_name || 'Unknown';
            }
            info.appendChild(name);

            var setDiv = document.createElement('div');
            setDiv.className = 'tile-set';
            setDiv.textContent = card.set_name || '';
            info.appendChild(setDiv);

            var price = document.createElement('div');
            price.className = 'tile-price';
            if (card.status === 'scanning' || card.status === 'pending') {
                price.innerHTML = '<span class="spinner-ring" style="width:14px;height:14px;border-width:2px"></span>';
            } else if (card.market_price) {
                price.textContent = '$' + Number(card.market_price).toFixed(2);
            } else {
                price.textContent = '--';
                price.style.color = 'var(--text-dim)';
            }
            info.appendChild(price);

            tile.appendChild(info);

            // Tap handler (only for resolved cards)
            if (card.status === 'resolved' || card.status === 'error') {
                (function(c) {
                    tile.addEventListener('click', function() { openModal(c); });
                })(card);
            }

            grid.appendChild(tile);
        }
    }

    function updateSummary() {
        var bar = document.getElementById('summaryBar');
        bar.classList.add('show');

        var total = 0;
        var resolved = 0;
        var pending = 0;

        for (var i = 0; i < cards.length; i++) {
            if (cards[i].status === 'resolved') {
                resolved++;
                if (cards[i].market_price) {
                    total += Number(cards[i].market_price);
                }
            } else {
                pending++;
            }
        }

        document.getElementById('summaryTotal').textContent = '$' + total.toFixed(2);
        document.getElementById('summaryCount').textContent = cards.length + ' card' + (cards.length !== 1 ? 's' : '');
        var statusEl = document.getElementById('summaryStatus');
        if (pending > 0) {
            statusEl.innerHTML = '<span class="spinner-ring" style="width:12px;height:12px;border-width:2px"></span> ' + pending + ' pending';
        } else {
            statusEl.textContent = resolved + ' identified';
        }
    }

    // ---- Detail Modal ----
    window.openModal = function(card) {
        var overlay = document.getElementById('detailModal');
        overlay.classList.add('show');

        var img = document.getElementById('modalImg');
        if (card.image_url) {
            img.src = card.image_url;
            img.style.display = 'block';
        } else {
            img.style.display = 'none';
        }

        document.getElementById('modalName').textContent = card.card_name || 'Unknown Card';
        document.getElementById('modalSet').textContent = card.set_name || '';
        document.getElementById('modalPrice').textContent = card.market_price ? '$' + Number(card.market_price).toFixed(2) : 'No price data';
        document.getElementById('modalCardId').textContent = card.card_id || '--';
        document.getElementById('modalMethod').textContent = card.method || '--';
        document.getElementById('modalConfidence').textContent = card.confidence ? (Math.round(card.confidence * 100) + '%') : '--';
    };

    window.closeModal = function() {
        document.getElementById('detailModal').classList.remove('show');
    };

    // Close modal on overlay click
    document.getElementById('detailModal').addEventListener('click', function(e) {
        if (e.target === this) closeModal();
    });

    // Swipe down to close modal
    (function() {
        var sheet = document.getElementById('modalSheet');
        var startY = 0;
        sheet.addEventListener('touchstart', function(e) {
            startY = e.touches[0].clientY;
        }, { passive: true });
        sheet.addEventListener('touchend', function(e) {
            var dy = e.changedTouches[0].clientY - startY;
            if (dy > 80) closeModal();
        }, { passive: true });
    })();

    // ---- Wire up file inputs ----
    document.getElementById('camera').onchange = function() { handleFile(this.files[0]); this.value = ''; };
    document.getElementById('gallery').onchange = function() { handleFile(this.files[0]); this.value = ''; };

})();
</script>
</body>
</html>
"""
