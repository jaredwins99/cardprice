"""Inventory management UI page.

Exports INVENTORY_HTML: a self-contained HTML page for viewing and managing
the card inventory with search, sort, quantity adjustments, and CSV export.
"""

INVENTORY_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=no">
<title>Card Inventory</title>
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
    padding: 8px 0 12px;
}
.header h1 {
    color: var(--accent);
    font-size: 22px;
    font-weight: 700;
}
.header .subtitle {
    color: var(--text-dim);
    font-size: 13px;
    margin-top: 2px;
}

/* Summary bar */
.summary {
    display: flex;
    justify-content: space-around;
    background: var(--bg-card);
    border-radius: var(--radius);
    padding: 14px 8px;
    margin-bottom: 12px;
}
.summary .stat {
    text-align: center;
}
.summary .stat .val {
    font-size: 22px;
    font-weight: 700;
    color: var(--green);
}
.summary .stat .lbl {
    font-size: 11px;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Controls row */
.controls {
    display: flex;
    gap: 8px;
    margin-bottom: 12px;
    flex-wrap: wrap;
}
.search-box {
    flex: 1;
    min-width: 140px;
    padding: 10px 12px;
    border-radius: var(--radius-sm);
    border: 1px solid var(--text-faint);
    background: var(--bg-card);
    color: var(--text);
    font-size: 14px;
    outline: none;
}
.search-box:focus {
    border-color: var(--accent);
}
.search-box::placeholder {
    color: var(--text-faint);
}
.sort-select {
    padding: 10px 12px;
    border-radius: var(--radius-sm);
    border: 1px solid var(--text-faint);
    background: var(--bg-card);
    color: var(--text);
    font-size: 13px;
    outline: none;
    cursor: pointer;
}
.sort-select:focus {
    border-color: var(--accent);
}
.export-btn {
    padding: 10px 16px;
    border-radius: var(--radius-sm);
    border: none;
    background: var(--accent);
    color: #fff;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    white-space: nowrap;
}
.export-btn:active {
    background: var(--accent-dark);
}

/* Card list */
.card-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.inv-card {
    background: var(--bg-card);
    border-radius: var(--radius);
    padding: 12px;
    transition: background 0.15s;
}
.inv-card:active {
    background: var(--bg-card-hover);
}

.card-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 8px;
}
.card-info {
    flex: 1;
    min-width: 0;
}
.card-name {
    font-size: 14px;
    font-weight: 600;
    color: var(--text);
    text-decoration: none;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    display: block;
}
a.card-name:hover {
    color: var(--accent);
}
.card-set {
    font-size: 11px;
    color: var(--text-dim);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-top: 2px;
}
.card-price {
    font-size: 16px;
    font-weight: 700;
    color: var(--green);
    white-space: nowrap;
}

/* Quantity controls */
.qty-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 8px;
}
.qty-controls {
    display: flex;
    align-items: center;
    gap: 0;
}
.qty-btn {
    width: 32px;
    height: 32px;
    border: 1px solid var(--text-faint);
    background: var(--bg-primary);
    color: var(--text);
    font-size: 18px;
    font-weight: 700;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    user-select: none;
    -webkit-user-select: none;
}
.qty-btn:active {
    background: var(--accent-dark);
}
.qty-btn.minus {
    border-radius: var(--radius-sm) 0 0 var(--radius-sm);
}
.qty-btn.plus {
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
}
.qty-val {
    width: 40px;
    height: 32px;
    border-top: 1px solid var(--text-faint);
    border-bottom: 1px solid var(--text-faint);
    border-left: none;
    border-right: none;
    background: var(--bg-card);
    color: var(--text);
    font-size: 14px;
    font-weight: 600;
    text-align: center;
    line-height: 32px;
}
.line-total {
    font-size: 13px;
    color: var(--text-dim);
}
.line-total .amount {
    color: var(--green-dim);
    font-weight: 600;
}

/* Condition prices row */
.cond-prices {
    display: flex;
    gap: 2px;
    margin-top: 6px;
    font-size: 10px;
    font-variant-numeric: tabular-nums;
}
.cond-prices .cp {
    flex: 1;
    text-align: center;
    padding: 3px 2px;
    border-radius: 4px;
    background: rgba(255,255,255,0.04);
}
.cond-prices .cp .cl { opacity: 0.5; font-size: 9px; display: block; }
.cond-prices .cp.nm { color: #4ecca3; }
.cond-prices .cp.lp { color: #a8d8a8; }
.cond-prices .cp.mp { color: #f1c40f; }
.cond-prices .cp.hp { color: #e67e22; }
.cond-prices .cp.dmg { color: #e74c3c; }
.cond-prices .cp.blank { color: var(--text-faint); }

/* Loading / empty states */
.loading {
    text-align: center;
    padding: 40px 0;
    color: var(--text-dim);
}
.spinner {
    display: inline-block;
    width: 28px;
    height: 28px;
    border: 3px solid var(--text-faint);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

.empty-state {
    text-align: center;
    padding: 60px 20px;
    color: var(--text-dim);
    font-size: 14px;
}

/* Nav link */
.nav-link {
    display: inline-block;
    color: var(--accent);
    text-decoration: none;
    font-size: 13px;
    margin-bottom: 8px;
}
.nav-link:hover { text-decoration: underline; }

/* Toast */
.toast {
    position: fixed;
    bottom: 20px;
    left: 50%;
    transform: translateX(-50%) translateY(80px);
    background: var(--bg-card);
    color: var(--text);
    padding: 10px 20px;
    border-radius: var(--radius-sm);
    font-size: 13px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    transition: transform 0.3s ease;
    z-index: 1000;
    pointer-events: none;
}
.toast.show {
    transform: translateX(-50%) translateY(0);
}
</style>
</head>
<body>
<div class="container">
    <a href="/" class="nav-link">&larr; Scanner</a>
    <div class="header">
        <h1>Inventory</h1>
        <div class="subtitle" id="lastUpdated"></div>
    </div>

    <div class="summary">
        <div class="stat">
            <div class="val" id="totalCards">-</div>
            <div class="lbl">Cards</div>
        </div>
        <div class="stat">
            <div class="val" id="uniqueCards">-</div>
            <div class="lbl">Unique</div>
        </div>
        <div class="stat">
            <div class="val" id="totalValue">-</div>
            <div class="lbl">NM Value</div>
        </div>
    </div>

    <div class="controls">
        <input type="text" class="search-box" id="searchInput" placeholder="Search cards...">
        <select class="sort-select" id="sortSelect">
            <option value="value-desc">Value (high)</option>
            <option value="value-asc">Value (low)</option>
            <option value="name-asc">Name (A-Z)</option>
            <option value="name-desc">Name (Z-A)</option>
            <option value="set-asc">Set (A-Z)</option>
            <option value="qty-desc">Qty (high)</option>
            <option value="qty-asc">Qty (low)</option>
        </select>
        <button class="export-btn" id="exportBtn">Export CSV</button>
    </div>

    <div id="loadingState" class="loading">
        <div class="spinner"></div>
        <div style="margin-top:10px;">Loading inventory...</div>
    </div>

    <div id="emptyState" class="empty-state" style="display:none;">
        No cards in inventory yet.<br>Scan some cards to get started!
    </div>

    <div class="card-list" id="cardList"></div>
</div>

<div class="toast" id="toast"></div>

<script>
var allItems = [];
var currentSort = 'value-desc';
var searchTerm = '';

function showToast(msg) {
    var t = document.getElementById('toast');
    t.textContent = msg;
    t.classList.add('show');
    setTimeout(function() { t.classList.remove('show'); }, 2000);
}

function formatPrice(p) {
    if (p == null) return '-';
    if (p >= 10) return '$' + Math.round(p);
    return '$' + p.toFixed(2);
}

function computeConditionPrices(nm) {
    if (!nm) return null;
    var mults = {
        NM: [1.0, 0.95, 1.05],
        LP: [0.80, 0.72, 0.88],
        MP: [0.60, 0.50, 0.70],
        HP: [0.40, 0.30, 0.50],
        DMG: [0.25, 0.15, 0.35]
    };
    var result = {};
    var conds = ['NM','LP','MP','HP','DMG'];
    for (var i = 0; i < conds.length; i++) {
        var c = conds[i];
        var m = mults[c];
        result[c] = {
            price: Math.round(nm * m[0] * 100) / 100,
            range_low: Math.round(nm * m[1] * 100) / 100,
            range_high: Math.round(nm * m[2] * 100) / 100
        };
    }
    return result;
}

function loadInventory() {
    fetch('/inventory')
        .then(function(r) { return r.json(); })
        .then(function(data) {
            document.getElementById('loadingState').style.display = 'none';
            allItems = data.items || [];
            if (allItems.length === 0) {
                document.getElementById('emptyState').style.display = '';
                return;
            }
            updateSummary();
            renderList();
        })
        .catch(function(e) {
            document.getElementById('loadingState').innerHTML =
                '<div style="color:var(--accent);">Failed to load inventory</div>';
        });
}

function updateSummary() {
    var totalQty = 0;
    var totalVal = 0;
    for (var i = 0; i < allItems.length; i++) {
        totalQty += allItems[i].quantity || 0;
        if (allItems[i].market_price) {
            totalVal += allItems[i].market_price * (allItems[i].quantity || 1);
        }
    }
    document.getElementById('totalCards').textContent = totalQty;
    document.getElementById('uniqueCards').textContent = allItems.length;
    document.getElementById('totalValue').textContent = formatPrice(totalVal);
}

function getSortedFiltered() {
    var items = allItems.slice();
    if (searchTerm) {
        var q = searchTerm.toLowerCase();
        items = items.filter(function(it) {
            return (it.name && it.name.toLowerCase().indexOf(q) >= 0) ||
                   (it.set_name && it.set_name.toLowerCase().indexOf(q) >= 0) ||
                   (it.card_id && it.card_id.toLowerCase().indexOf(q) >= 0);
        });
    }
    items.sort(function(a, b) {
        switch (currentSort) {
            case 'value-desc':
                return ((b.market_price || 0) * (b.quantity || 1)) - ((a.market_price || 0) * (a.quantity || 1));
            case 'value-asc':
                return ((a.market_price || 0) * (a.quantity || 1)) - ((b.market_price || 0) * (b.quantity || 1));
            case 'name-asc':
                return (a.name || '').localeCompare(b.name || '');
            case 'name-desc':
                return (b.name || '').localeCompare(a.name || '');
            case 'set-asc':
                return (a.set_name || a.set_id || '').localeCompare(b.set_name || b.set_id || '');
            case 'qty-desc':
                return (b.quantity || 0) - (a.quantity || 0);
            case 'qty-asc':
                return (a.quantity || 0) - (b.quantity || 0);
            default:
                return 0;
        }
    });
    return items;
}

function renderList() {
    var items = getSortedFiltered();
    var list = document.getElementById('cardList');
    var html = '';
    for (var i = 0; i < items.length; i++) {
        var it = items[i];
        var nm = it.market_price;
        var lineTotal = nm ? nm * (it.quantity || 1) : null;
        var conds = computeConditionPrices(nm);
        var tcgUrl = it.tcgplayer_url || null;

        html += '<div class="inv-card" data-idx="' + i + '" data-cid="' + (it.card_id || '') + '">';
        html += '<div class="card-top">';
        html += '<div class="card-info">';
        if (tcgUrl) {
            html += '<a class="card-name" href="' + tcgUrl + '" target="_blank" rel="noopener">' + escHtml(it.name || it.card_id) + '</a>';
        } else {
            html += '<span class="card-name">' + escHtml(it.name || it.card_id) + '</span>';
        }
        html += '<div class="card-set">' + escHtml(it.set_name || it.set_id || '') + '</div>';
        html += '</div>';
        html += '<div class="card-price">' + (nm ? formatPrice(nm) : '<span style="color:var(--text-faint)">-</span>') + '</div>';
        html += '</div>';

        // Quantity row
        html += '<div class="qty-row">';
        html += '<div class="qty-controls">';
        html += '<button class="qty-btn minus" onclick="adjustQty(\'' + escAttr(it.card_id) + '\', -1, this)">-</button>';
        html += '<div class="qty-val" id="qty-' + escAttr(it.card_id) + '">' + (it.quantity || 1) + '</div>';
        html += '<button class="qty-btn plus" onclick="adjustQty(\'' + escAttr(it.card_id) + '\', 1, this)">+</button>';
        html += '</div>';
        html += '<div class="line-total">';
        if (lineTotal != null) {
            html += 'Total: <span class="amount">' + formatPrice(lineTotal) + '</span>';
        }
        html += '</div>';
        html += '</div>';

        // Condition prices
        if (conds) {
            html += '<div class="cond-prices">';
            var condKeys = ['NM','LP','MP','HP','DMG'];
            var condCls = ['nm','lp','mp','hp','dmg'];
            for (var ci = 0; ci < condKeys.length; ci++) {
                var cp = conds[condKeys[ci]];
                html += '<div class="cp ' + condCls[ci] + '">';
                html += '<span class="cl">' + condKeys[ci] + '</span>';
                html += formatPrice(cp.price);
                html += '</div>';
            }
            html += '</div>';
        }

        html += '</div>';
    }
    list.innerHTML = html;

    // Show empty state for filter
    if (items.length === 0 && allItems.length > 0) {
        list.innerHTML = '<div class="empty-state">No cards match your search.</div>';
    }
}

function escHtml(s) {
    var d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
}

function escAttr(s) {
    return (s || '').replace(/'/g, "\\'").replace(/"/g, '&quot;');
}

function adjustQty(cardId, delta, btn) {
    btn.disabled = true;
    var endpoint = delta > 0 ? '/inventory/add' : '/inventory/remove';
    var qty = Math.abs(delta);

    fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ card_id: cardId, quantity: qty })
    })
    .then(function(r) { return r.json(); })
    .then(function(data) {
        btn.disabled = false;
        if (data.error) {
            showToast('Error: ' + data.error);
            return;
        }
        var newQty = data.quantity;
        if (newQty <= 0) {
            // Remove from allItems
            for (var i = 0; i < allItems.length; i++) {
                if (allItems[i].card_id === cardId) {
                    allItems.splice(i, 1);
                    break;
                }
            }
            showToast('Removed from inventory');
            if (allItems.length === 0) {
                document.getElementById('emptyState').style.display = '';
            }
        } else {
            // Update quantity
            for (var i = 0; i < allItems.length; i++) {
                if (allItems[i].card_id === cardId) {
                    allItems[i].quantity = newQty;
                    break;
                }
            }
            showToast(delta > 0 ? 'Added! Qty: ' + newQty : 'Qty: ' + newQty);
        }
        updateSummary();
        renderList();
    })
    .catch(function(e) {
        btn.disabled = false;
        showToast('Network error');
    });
}

// Event listeners
document.getElementById('searchInput').addEventListener('input', function() {
    searchTerm = this.value;
    renderList();
});

document.getElementById('sortSelect').addEventListener('change', function() {
    currentSort = this.value;
    renderList();
});

document.getElementById('exportBtn').addEventListener('click', function() {
    window.location.href = '/export';
});

// Load on page ready
loadInventory();
</script>
</body>
</html>
"""
