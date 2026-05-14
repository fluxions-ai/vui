"""HTML frontend for the streaming TTS server. Served from index.html file."""

from pathlib import Path

FRONTEND_DIR = Path(__file__).parent
INDEX_HTML = FRONTEND_DIR / "index.html"
VISUALIZER_JS = FRONTEND_DIR / "visualizer.js"


def get_html() -> str:
    return INDEX_HTML.read_text()


def get_visualizer_js() -> str:
    return VISUALIZER_JS.read_text()


VISUALIZER_DEMO_HTML = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Vui Voice Visualizer Demo</title>
<style>
  body { background:#0a0a0a; color:#ddd; font-family:system-ui,sans-serif;
         display:flex; flex-direction:column; align-items:center;
         justify-content:center; min-height:100vh; margin:0; gap:24px; }
  #orb { width:340px; height:340px; }
  .row { display:flex; gap:8px; flex-wrap:wrap; justify-content:center; }
  button { background:#222; color:#eee; border:1px solid #444; border-radius:6px;
           padding:8px 14px; cursor:pointer; font-size:0.9em; }
  button:hover { background:#333; }
  code { background:#1a1a1a; padding:2px 6px; border-radius:3px; font-size:0.85em; }
  .hint { color:#888; font-size:0.85em; max-width:520px; text-align:center; line-height:1.5; }
</style></head>
<body>
  <div id="orb"></div>
  <div class="row">
    <button onclick="viz.connect('')">Connect to this server</button>
    <button onclick="viz.setState('idle')">idle</button>
    <button onclick="viz.setState('listening')">listening</button>
    <button onclick="viz.setState('thinking')">thinking</button>
    <button onclick="viz.setState('speaking')">speaking</button>
  </div>
  <div class="hint">
    Drop into any frontend with one tag:<br>
    <code>&lt;script src="/visualizer.js"&gt;&lt;/script&gt;</code><br>
    <code>&lt;vui-voice-viz src="ws://localhost:8080"&gt;&lt;/vui-voice-viz&gt;</code>
  </div>
  <script src="/visualizer.js"></script>
  <script>
    const viz = new VuiVoiceVisualizer({ container: document.getElementById('orb') });
  </script>
</body></html>
"""


def get_visualizer_demo_html() -> str:
    return VISUALIZER_DEMO_HTML
