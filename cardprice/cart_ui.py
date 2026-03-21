"""Shopping cart UI page.

Exports CART_HTML: a self-contained HTML page for viewing and managing
a shopping cart with quantity adjustments, condition pricing, and
inventory comparison.
"""

CART_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=no">
<title>Shopping Cart</title>
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
    --blue: #5dade2;
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
.clear-btn {
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
.clear-btn:active {
    background: var(--accent-dark);
}

/* Card list */
.card-list {
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.cart-card {
    background: var(--bg-card);
    border-radius: var(--radius);
    padding: 12px;
    transition: background 0.15s;
}
.cart-card:active {
    background: var(--bg-card-hover);
}
.cart-card.in-inventory {
    border-left: 3px solid var(--blue);
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
.inv-badge {
    font-size: 10px;
    color: var(--blue);
    margin-top: 2px;
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
.remove-btn {
    background: none;
    border: 1px solid var(--accent);
    color: var(--accent);
    padding: 4px 10px;
    border-radius: var(--radius-sm);
    font-size: 12px;
    cursor: pointer;
    margin-left: 12px;
}
.remove-btn:active {
    background: var(--accent);
    color: #fff;
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

/* Compare to inventory section */
.compare-section {
    margin-top: 16px;
    background: var(--bg-card);
    border-radius: var(--radius);
    padding: 12px;
}
.compare-section h3 {
    font-size: 14px;
    color: var(--blue);
    margin-bottom: 8px;
}
.compare-item {
    display: flex;
    justify-content: space-between;
    padding: 4px 0;
    font-size: 12px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.compare-item:last-child { border-bottom: none; }
.compare-item .name { color: var(--text); }
.compare-item .qty-info { color: var(--text-dim); }
.compare-item .qty-info .owned { color: var(--blue); }
.compare-empty {
    color: var(--text-faint);
    font-size: 12px;
}

/* Total bar */
.total-bar {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: var(--bg-card);
    border-top: 1px solid var(--text-faint);
    padding: 12px 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    z-index: 100;
}
.total-bar .total-label {
    font-size: 14px;
    color: var(--text-dim);
}
.total-bar .total-amount {
    font-size: 22px;
    font-weight: 700;
    color: var(--green);
}

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
    bottom: 70px;
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
    <span style="margin:0 6px;color:var(--text-faint)">|</span>
    <a href="/inventory/view" class="nav-link">Inventory</a>

    <div class="header">
        <h1>Shopping Cart</h1>
        <div class="subtitle">Cards you want to buy</div>
    </div>

    <div class="summary">
        <div class="stat">
            <div class="val" id="totalItems">-</div>
            <div class="lbl">Items</div>
        </div>
        <div class="stat">
            <div class="val" id="uniqueItems">-</div>
            <div class="lbl">Unique</div>
        </div>
        <div class="stat">
            <div class="val" id="totalValue">-</div>
            <div class="lbl">Est. Value</div>
        </div>
    </div>

    <div class="controls">
        <input type="text" class="search-box" id="searchInput" placeholder="Search cart...">
        <select class="sort-select" id="sortSelect">
            <option value="value-desc">Value (high)</option>
            <option value="value-asc">Value (low)</option>
            <option value="name-asc">Name (A-Z)</option>
            <option value="name-desc">Name (Z-A)</option>
            <option value="set-asc">Set (A-Z)</option>
            <option value="qty-desc">Qty (high)</option>
        </select>
        <button class="clear-btn" id="clearBtn">Clear Cart</button>
    </div>

    <div id="loadingState" class="loading">
        <div class="spinner"></div>
        <div style="margin-top:10px;">Loading cart...</div>
    </div>

    <div id="emptyState" class="empty-state" style="display:none;">
        Your cart is empty.<br>Scan cards and add them to your cart!
    </div>

    <div class="card-list" id="cardList"></div>

    <div id="compareSection" class="compare-section" style="display:none;">
        <h3>Already in Inventory</h3>
        <div id="compareList"></div>
    </div>
</div>

<div class="total-bar" id="totalBar" style="display:none;">
    <div class="total-label">Cart Total</div>
    <div class="total-amount" id="cartTotal">$0</div>
</div>

<div class="toast" id="toast"></div>

<script>
var cartItems = [];
var inventoryItems = [];
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

function loadCart() {
    fetch('/cart')
        .then(function(r) { return r.json(); })
        .then(function(data) {
            document.getElementById('loadingState').style.display = 'none';
            cartItems = data.items || [];
            if (cartItems.length === 0) {
                document.getElementById('emptyState').style.display = '';
                document.getElementById('totalBar').style.display = 'none';
                updateSummary();
                return;
            }
            document.getElementById('totalBar').style.display = '';
            updateSummary();
            renderList();
            loadInventoryForComparison();
        })
        .catch(function(e) {
            document.getElementById('loadingState').innerHTML =
                '<div style="color:var(--accent);">Failed to load cart</div>';
        });
}

function loadInventoryForComparison() {
    fetch('/inventory')
        .then(function(r) { return r.json(); })
        .then(function(data) {
            inventoryItems = data.items || [];
            renderCompareSection();
        })
        .catch(function() {
            inventoryItems = [];
        });
}

function updateSummary() {
    var totalQty = 0;
    var totalVal = 0;
    for (var i = 0; i < cartItems.length; i++) {
        var qty = cartItems[i].quantity || 1;
        totalQty += qty;
        if (cartItems[i].market_price) {
            totalVal += cartItems[i].market_price * qty;
        }
    }
    document.getElementById('totalItems').textContent = totalQty;
    document.getElementById('uniqueItems').textContent = cartItems.length;
    document.getElementById('totalValue').textContent = formatPrice(totalVal);
    document.getElementById('cartTotal').textContent = formatPrice(totalVal);
}

function getSortedFiltered() {
    var items = cartItems.slice();
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
                return (a.set_name || '').localeCompare(b.set_name || '');
            case 'qty-desc':
                return (b.quantity || 0) - (a.quantity || 0);
            default:
                return 0;
        }
    });
    return items;
}

function getInventoryMap() {
    var map = {};
    for (var i = 0; i < inventoryItems.length; i++) {
        map[inventoryItems[i].card_id] = inventoryItems[i];
    }
    return map;
}

function renderList() {
    var items = getSortedFiltered();
    var invMap = getInventoryMap();
    var list = document.getElementById('cardList');
    var html = '';
    for (var i = 0; i < items.length; i++) {
        var it = items[i];
        var nm = it.market_price;
        var qty = it.quantity || 1;
        var lineTotal = nm ? nm * qty : null;
        var conds = computeConditionPrices(nm);
        var tcgUrl = it.tcgplayer_url || null;
        var inInv = invMap[it.card_id];
        var cardCls = 'cart-card' + (inInv ? ' in-inventory' : '');

        html += '<div class="' + cardCls + '">';
        html += '<div class="card-top">';
        html += '<div class="card-info">';
        if (tcgUrl) {
            html += '<a class="card-name" href="' + tcgUrl + '" target="_blank" rel="noopener">' + escHtml(it.name || it.card_id) + '</a>';
        } else {
            html += '<span class="card-name">' + escHtml(it.name || it.card_id) + '</span>';
        }
        html += '<div class="card-set">' + escHtml(it.set_name || '') + '</div>';
        if (inInv) {
            html += '<div class="inv-badge">Owned: ' + (inInv.quantity || 1) + ' in inventory</div>';
        }
        html += '</div>';
        html += '<div class="card-price">' + (nm ? formatPrice(nm) : '<span style="color:var(--text-faint)">-</span>') + '</div>';
        html += '</div>';

        // Quantity row
        html += '<div class="qty-row">';
        html += '<div class="qty-controls">';
        html += '<button class="qty-btn minus" onclick="adjustQty(\'' + escAttr(it.card_id) + '\', -1, this)">-</button>';
        html += '<div class="qty-val">' + qty + '</div>';
        html += '<button class="qty-btn plus" onclick="adjustQty(\'' + escAttr(it.card_id) + '\', 1, this)">+</button>';
        html += '<button class="remove-btn" onclick="removeItem(\'' + escAttr(it.card_id) + '\', this)">Remove</button>';
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

    if (items.length === 0 && cartItems.length > 0) {
        list.innerHTML = '<div class="empty-state">No cards match your search.</div>';
    }
}

function renderCompareSection() {
    var section = document.getElementById('compareSection');
    var compareList = document.getElementById('compareList');
    var invMap = getInventoryMap();
    var matches = [];

    for (var i = 0; i < cartItems.length; i++) {
        var inv = invMap[cartItems[i].card_id];
        if (inv) {
            matches.push({
                name: cartItems[i].name || cartItems[i].card_id,
                cartQty: cartItems[i].quantity || 1,
                invQty: inv.quantity || 1
            });
        }
    }

    if (matches.length === 0) {
        section.style.display = 'none';
        return;
    }

    section.style.display = '';
    var html = '';
    for (var j = 0; j < matches.length; j++) {
        var m = matches[j];
        html += '<div class="compare-item">';
        html += '<span class="name">' + escHtml(m.name) + '</span>';
        html += '<span class="qty-info">Cart: ' + m.cartQty + ' / <span class="owned">Owned: ' + m.invQty + '</span></span>';
        html += '</div>';
    }
    compareList.innerHTML = html;
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
    var endpoint = delta > 0 ? '/cart/add' : '/cart/remove';

    fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ card_id: cardId, quantity: 1 })
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
            for (var i = 0; i < cartItems.length; i++) {
                if (cartItems[i].card_id === cardId) {
                    cartItems.splice(i, 1);
                    break;
                }
            }
            showToast('Removed from cart');
            if (cartItems.length === 0) {
                document.getElementById('emptyState').style.display = '';
                document.getElementById('totalBar').style.display = 'none';
                document.getElementById('compareSection').style.display = 'none';
            }
        } else {
            for (var i = 0; i < cartItems.length; i++) {
                if (cartItems[i].card_id === cardId) {
                    cartItems[i].quantity = newQty;
                    break;
                }
            }
            showToast(delta > 0 ? 'Added! Qty: ' + newQty : 'Qty: ' + newQty);
        }
        updateSummary();
        renderList();
        renderCompareSection();
    })
    .catch(function(e) {
        btn.disabled = false;
        showToast('Network error');
    });
}

function removeItem(cardId, btn) {
    btn.disabled = true;

    fetch('/cart/remove', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ card_id: cardId, quantity: 9999 })
    })
    .then(function(r) { return r.json(); })
    .then(function(data) {
        btn.disabled = false;
        for (var i = 0; i < cartItems.length; i++) {
            if (cartItems[i].card_id === cardId) {
                cartItems.splice(i, 1);
                break;
            }
        }
        showToast('Removed from cart');
        if (cartItems.length === 0) {
            document.getElementById('emptyState').style.display = '';
            document.getElementById('totalBar').style.display = 'none';
            document.getElementById('compareSection').style.display = 'none';
            document.getElementById('cardList').innerHTML = '';
        }
        updateSummary();
        renderList();
        renderCompareSection();
    })
    .catch(function(e) {
        btn.disabled = false;
        showToast('Network error');
    });
}

function clearCart() {
    if (!confirm('Clear all items from cart?')) return;

    fetch('/cart/clear')
        .then(function(r) { return r.json(); })
        .then(function(data) {
            cartItems = [];
            document.getElementById('emptyState').style.display = '';
            document.getElementById('totalBar').style.display = 'none';
            document.getElementById('compareSection').style.display = 'none';
            document.getElementById('cardList').innerHTML = '';
            updateSummary();
            showToast('Cart cleared');
        })
        .catch(function(e) {
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

document.getElementById('clearBtn').addEventListener('click', clearCart);

// Load on page ready
loadCart();
</script>
</body>
</html>
"""
