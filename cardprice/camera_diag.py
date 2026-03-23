"""Camera diagnostic page - tests every step of the camera pipeline."""

CAMERA_DIAG_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Camera Diagnostics</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, system-ui, sans-serif; background: #111; color: #eee; padding: 16px; }
h1 { font-size: 20px; margin-bottom: 16px; }
.step { display: flex; align-items: flex-start; gap: 8px; padding: 6px 0; font-size: 14px; font-family: monospace; }
.step .icon { flex-shrink: 0; width: 20px; text-align: center; }
.step.pass .icon { color: #4caf50; }
.step.fail .icon { color: #f44336; }
.step.pending .icon { color: #666; }
.step .label { flex: 1; }
.step.fail .label { color: #f44336; }
.troubleshoot { background: #2a1a1a; border: 1px solid #f44336; border-radius: 6px; padding: 10px 12px; margin: 6px 0 6px 28px; font-size: 13px; color: #ff8a80; }
#camera-feed { display: none; margin-top: 16px; width: 100%; max-width: 640px; border-radius: 8px; }
#success-msg { display: none; margin-top: 12px; padding: 12px; background: #1b3a1b; border: 1px solid #4caf50; border-radius: 8px; color: #81c784; font-size: 16px; font-weight: 600; text-align: center; }
</style>
</head>
<body>
<h1>Camera Diagnostics</h1>
<div id="steps"></div>
<div id="success-msg">Camera works!</div>
<video id="camera-feed" autoplay playsinline muted></video>

<script>
const stepsEl = document.getElementById('steps');
const video = document.getElementById('camera-feed');
let stream = null;
let allPassed = true;

const TROUBLESHOOT = {
    'https': 'Need HTTPS. Use the tunnel URL (check /tunnel-url).',
    'mediaDevices': 'navigator.mediaDevices is undefined. Need HTTPS or localhost.',
    'getUserMedia': 'getUserMedia not available. Browser too old or not HTTPS.',
    'permission': 'Tap "Allow" when prompted. If dismissed, reset site permissions in browser settings.',
    'permission_denied': 'Permission denied. Go to browser Settings > Site Settings > Camera and allow this site.',
    'stream': 'No video tracks in stream. Camera may be in use by another app.',
    'black_frame': 'Check Brave shields, try disabling fingerprinting protection. On iOS, try Safari instead of Chrome.',
    'play': 'Video play() failed. Try tapping the page first (autoplay policy).',
    'no_camera': 'No camera found. Check that a camera is connected and not blocked by OS permissions.',
};

function addStep(id) {
    const div = document.createElement('div');
    div.className = 'step pending';
    div.id = 'step-' + id;
    div.innerHTML = '<span class="icon">&#9675;</span><span class="label">...</span>';
    stepsEl.appendChild(div);
    return div;
}

function passStep(id, text) {
    const el = document.getElementById('step-' + id);
    el.className = 'step pass';
    el.querySelector('.icon').innerHTML = '&#10003;';
    el.querySelector('.label').textContent = text;
}

function failStep(id, text, troubleshootKey) {
    allPassed = false;
    const el = document.getElementById('step-' + id);
    el.className = 'step fail';
    el.querySelector('.icon').innerHTML = '&#10007;';
    el.querySelector('.label').textContent = text;
    if (troubleshootKey && TROUBLESHOOT[troubleshootKey]) {
        const tip = document.createElement('div');
        tip.className = 'troubleshoot';
        tip.textContent = TROUBLESHOOT[troubleshootKey];
        el.after(tip);
    }
}

// Pre-create all step placeholders
const stepIds = [
    'https', 'mediaDevices', 'getUserMedia', 'permission',
    'stream', 'trackSettings', 'videoElement', 'srcObject',
    'loadedmetadata', 'loadeddata', 'play', 'dimensions',
    'paused', 'readyState', 'frameCapture'
];
stepIds.forEach(addStep);

async function runDiagnostics() {
    // Step 1: HTTPS
    const isHTTPS = location.protocol === 'https:' || location.hostname === 'localhost' || location.hostname === '127.0.0.1';
    if (isHTTPS) {
        passStep('https', 'HTTPS detected: ' + location.protocol + '//' + location.hostname);
    } else {
        failStep('https', 'HTTPS detected: NO (' + location.protocol + '//' + location.hostname + ')', 'https');
        return;
    }

    // Step 2: navigator.mediaDevices
    if (navigator.mediaDevices) {
        passStep('mediaDevices', 'navigator.mediaDevices exists: yes');
    } else {
        failStep('mediaDevices', 'navigator.mediaDevices exists: no (undefined)', 'mediaDevices');
        return;
    }

    // Step 3: getUserMedia
    if (typeof navigator.mediaDevices.getUserMedia === 'function') {
        passStep('getUserMedia', 'getUserMedia available: yes');
    } else {
        failStep('getUserMedia', 'getUserMedia available: no', 'getUserMedia');
        return;
    }

    // Step 4: Camera permission
    try {
        let permState = 'unknown';
        if (navigator.permissions && navigator.permissions.query) {
            try {
                const perm = await navigator.permissions.query({ name: 'camera' });
                permState = perm.state;
            } catch(e) {
                permState = 'query unsupported';
            }
        }
        passStep('permission', 'Camera permission: ' + permState + ' (will prompt if needed)');
    } catch(e) {
        passStep('permission', 'Camera permission: query unavailable (will prompt)');
    }

    // Step 5: Get stream
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment', width: { ideal: 1920 }, height: { ideal: 1080 } },
            audio: false
        });
    } catch(e) {
        const key = e.name === 'NotAllowedError' ? 'permission_denied' :
                     e.name === 'NotFoundError' ? 'no_camera' : 'stream';
        failStep('permission', 'Camera permission: ' + e.name + ' - ' + e.message, key);
        failStep('stream', 'Stream obtained: FAILED', key);
        return;
    }

    // Update permission step since we got through
    passStep('permission', 'Camera permission: granted');

    const videoTracks = stream.getVideoTracks();
    if (videoTracks.length > 0) {
        passStep('stream', 'Stream obtained: ' + videoTracks.length + ' video track(s), label: ' + videoTracks[0].label);
    } else {
        failStep('stream', 'Stream obtained: 0 video tracks', 'stream');
        return;
    }

    // Step 6: Track settings
    const settings = videoTracks[0].getSettings();
    const w = settings.width || '?';
    const h = settings.height || '?';
    const fps = settings.frameRate ? Math.round(settings.frameRate) : '?';
    const facing = settings.facingMode || 'unknown';
    passStep('trackSettings', 'Video track settings: ' + w + 'x' + h + ', ' + fps + 'fps, ' + facing);

    // Step 7: Video element
    passStep('videoElement', 'Video element: <video autoplay=' + video.autoplay + ' playsinline=' + video.hasAttribute('playsinline') + ' muted=' + video.muted + '>');

    // Step 8: Set srcObject
    video.srcObject = stream;
    passStep('srcObject', 'srcObject set: MediaStream active=' + stream.active);

    // Step 9: Wait for loadedmetadata
    try {
        await new Promise((resolve, reject) => {
            const timer = setTimeout(() => reject(new Error('timeout after 5s')), 5000);
            video.addEventListener('loadedmetadata', () => { clearTimeout(timer); resolve(); }, { once: true });
        });
        passStep('loadedmetadata', 'loadedmetadata fired: ' + video.videoWidth + 'x' + video.videoHeight);
    } catch(e) {
        failStep('loadedmetadata', 'loadedmetadata: ' + e.message, 'play');
        return;
    }

    // Step 10: Wait for loadeddata
    try {
        await new Promise((resolve, reject) => {
            const timer = setTimeout(() => reject(new Error('timeout after 5s')), 5000);
            if (video.readyState >= 2) { clearTimeout(timer); resolve(); return; }
            video.addEventListener('loadeddata', () => { clearTimeout(timer); resolve(); }, { once: true });
        });
        passStep('loadeddata', 'loadeddata fired: yes');
    } catch(e) {
        failStep('loadeddata', 'loadeddata: ' + e.message, 'play');
        return;
    }

    // Step 11: play()
    try {
        await video.play();
        passStep('play', 'play() succeeded: yes');
    } catch(e) {
        failStep('play', 'play() failed: ' + e.name + ' - ' + e.message, 'play');
        return;
    }

    // Step 12: Dimensions
    if (video.videoWidth > 0 && video.videoHeight > 0) {
        passStep('dimensions', 'Video dimensions: ' + video.videoWidth + 'x' + video.videoHeight);
    } else {
        failStep('dimensions', 'Video dimensions: ' + video.videoWidth + 'x' + video.videoHeight + ' (zero!)', 'black_frame');
    }

    // Step 13: Paused state
    if (!video.paused) {
        passStep('paused', 'Video paused: false');
    } else {
        failStep('paused', 'Video paused: true', 'play');
    }

    // Step 14: readyState
    const stateNames = ['HAVE_NOTHING', 'HAVE_METADATA', 'HAVE_CURRENT_DATA', 'HAVE_FUTURE_DATA', 'HAVE_ENOUGH_DATA'];
    const rs = video.readyState;
    if (rs >= 3) {
        passStep('readyState', 'Video readyState: ' + rs + ' (' + (stateNames[rs] || '?') + ')');
    } else {
        failStep('readyState', 'Video readyState: ' + rs + ' (' + (stateNames[rs] || '?') + ') - expected >= 3', 'black_frame');
    }

    // Step 15: Frame capture test
    try {
        // Wait a moment for frames to flow
        await new Promise(r => setTimeout(r, 500));
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0);
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const pixels = imageData.data;
        let nonBlack = 0;
        const totalPixels = canvas.width * canvas.height;
        // Sample every 100th pixel for speed
        for (let i = 0; i < pixels.length; i += 400) {
            const r = pixels[i], g = pixels[i+1], b = pixels[i+2];
            if (r > 10 || g > 10 || b > 10) nonBlack++;
        }
        const sampled = Math.floor(pixels.length / 400);
        const pct = Math.round(100 * nonBlack / sampled);
        if (pct > 5) {
            passStep('frameCapture', 'Frame capture test: ' + canvas.width + 'x' + canvas.height + ' canvas, non-black pixels: ' + pct + '%');
        } else {
            failStep('frameCapture', 'Frame capture test: ' + canvas.width + 'x' + canvas.height + ' canvas, non-black pixels: ' + pct + '% (frame is black!)', 'black_frame');
        }
    } catch(e) {
        failStep('frameCapture', 'Frame capture test failed: ' + e.message, 'black_frame');
    }

    // Show camera feed if all passed
    if (allPassed) {
        document.getElementById('success-msg').style.display = 'block';
        video.style.display = 'block';
    }
}

runDiagnostics();
</script>
</body>
</html>
"""
