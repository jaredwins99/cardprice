CAMERA_TEST_HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Camera Test</title>
<style>
body { margin: 0; background: #000; color: #fff; font-family: sans-serif; text-align: center; }
video { width: 100%; max-height: 60vh; }
button { font-size: 24px; padding: 20px 40px; margin: 20px; background: #4ecca3; border: none; border-radius: 12px; color: #000; }
img { width: 100%; max-height: 30vh; }
#status { padding: 10px; color: #f1c40f; }
</style>
</head>
<body>
<div id="status">Initializing camera...</div>
<video id="vid" autoplay playsinline muted></video>
<br>
<button id="btn" onclick="capture()">Capture</button>
<br>
<canvas id="can" style="display:none"></canvas>
<img id="img">
<script>
const vid = document.getElementById('vid');
const can = document.getElementById('can');
const img = document.getElementById('img');
const status = document.getElementById('status');

async function start() {
    try {
        if (!navigator.mediaDevices) {
            status.textContent = 'ERROR: navigator.mediaDevices is undefined. Need HTTPS.';
            return;
        }
        status.textContent = 'Requesting camera...';
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment' },
            audio: false
        });
        status.textContent = 'Got stream, setting up video...';
        vid.srcObject = stream;
        vid.onloadedmetadata = () => {
            status.textContent = 'Metadata loaded: ' + vid.videoWidth + 'x' + vid.videoHeight;
        };
        vid.onloadeddata = () => {
            status.textContent = 'Camera ready! ' + vid.videoWidth + 'x' + vid.videoHeight;
        };
        vid.onerror = (e) => {
            status.textContent = 'Video error: ' + e.message;
        };
        vid.play().catch(e => {
            status.textContent = 'Play failed: ' + e.message + ' (tap video to start)';
            vid.onclick = () => { vid.play(); status.textContent = 'Playing...'; };
        });
    } catch (e) {
        status.textContent = 'Camera error: ' + e.name + ': ' + e.message;
    }
}

function capture() {
    can.width = vid.videoWidth;
    can.height = vid.videoHeight;
    can.getContext('2d').drawImage(vid, 0, 0);
    img.src = can.toDataURL('image/jpeg', 0.8);
    status.textContent = 'Captured! ' + vid.videoWidth + 'x' + vid.videoHeight;
}

start();
</script>
</body>
</html>"""
