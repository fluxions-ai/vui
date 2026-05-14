// VuiVoiceVisualizer — drop-in reactive orb for vui streaming server.
//
// Visualizes three states: idle (breathing grey), listening (cyan, reactive
// to mic level), speaking (warm orange, reactive to assistant audio).
// State transitions are colour-lerped, no hard cuts.
//
// Quick start (vanilla):
//   <div id="orb" style="width:280px;height:280px"></div>
//   <script src="/visualizer.js"></script>
//   <script>
//     const viz = new VuiVoiceVisualizer({ container: document.getElementById('orb') });
//     viz.connect('ws://localhost:8080');  // grabs mic, opens WS+WebRTC
//   </script>
//
// Web component:
//   <vui-voice-viz src="ws://localhost:8080"></vui-voice-viz>
//
// Manual driving (no server):
//   const viz = new VuiVoiceVisualizer({ container: el });
//   viz.setState('speaking');
//   viz.setLevel(0.6);
//
// Wiring into an existing client (when you already have ws + streams):
//   viz.attachMicStream(myMicMediaStream);
//   viz.attachAssistantStream(remoteWebRtcStream);
//   viz.attachWebSocket(myWs);   // listens for vad_start/stop, reply, turn_done
//
// All attachments are optional; supply only what you have.

(function (root, factory) {
    if (typeof module === 'object' && module.exports) module.exports = factory();
    else root.VuiVoiceVisualizer = factory();
}(typeof self !== 'undefined' ? self : this, function () {

    const THEMES = {
        idle:      { hue: 290, sat: 20, light: 55 },  // muted lilac
        listening: { hue: 320, sat: 85, light: 65 },  // hot pink
        speaking:  { hue: 275, sat: 80, light: 62 },  // vivid purple
        thinking:  { hue: 305, sat: 75, light: 60 },  // magenta in between
    };

    function lerp(a, b, t) { return a + (b - a) * t; }
    function clamp01(x) { return x < 0 ? 0 : x > 1 ? 1 : x; }

    class VuiVoiceVisualizer {
        constructor(opts = {}) {
            this.container = opts.container;
            if (!this.container) throw new Error('VuiVoiceVisualizer: container required');
            this.opts = {
                size: opts.size || null,        // null = fill container
                background: opts.background ?? 'transparent',
                glow: opts.glow ?? true,
                rings: opts.rings ?? 3,
                fps: opts.fps ?? 60,
            };

            this._state = 'idle';
            this._targetTheme = { ...THEMES.idle };
            this._curTheme = { ...THEMES.idle };

            this._micLevel = 0;          // 0..1 smoothed
            this._asstLevel = 0;         // 0..1 smoothed
            this._driveLevel = 0;        // smoothed combined drive
            this._breath = 0;            // idle pulsation phase
            this._t0 = performance.now();
            this._destroyed = false;

            this._micAnalyser = null;
            this._asstAnalyser = null;
            this._micCtx = null;
            this._asstCtx = null;
            this._micBuf = null;
            this._asstBuf = null;

            this._ws = null;
            this._ownsWs = false;
            this._pc = null;
            this._micStream = null;
            this._remoteAudio = null;
            this._reconnectTimer = null;

            this._buildDOM();
            this._loop = this._loop.bind(this);
            requestAnimationFrame(this._loop);
        }

        // ---------- public API ----------

        setState(state) {
            if (!THEMES[state]) return;
            this._state = state;
            this._targetTheme = { ...THEMES[state] };
        }

        getState() { return this._state; }

        setLevel(level, source = 'auto') {
            level = clamp01(level);
            if (source === 'mic') this._micLevel = level;
            else if (source === 'assistant') this._asstLevel = level;
            else this._driveLevel = level;
        }

        attachMicStream(stream) {
            this._detachMic();
            if (!stream) return;
            this._micStream = stream;
            const ctx = new (window.AudioContext || window.webkitAudioContext)();
            const src = ctx.createMediaStreamSource(stream);
            const an = ctx.createAnalyser();
            an.fftSize = 512;
            an.smoothingTimeConstant = 0.6;
            src.connect(an);
            this._micCtx = ctx;
            this._micAnalyser = an;
            this._micBuf = new Uint8Array(an.frequencyBinCount);
        }

        attachAssistantStream(stream) {
            this._detachAssistant();
            if (!stream) return;
            const ctx = new (window.AudioContext || window.webkitAudioContext)();
            const src = ctx.createMediaStreamSource(stream);
            const an = ctx.createAnalyser();
            an.fftSize = 512;
            an.smoothingTimeConstant = 0.5;
            src.connect(an);
            this._asstCtx = ctx;
            this._asstAnalyser = an;
            this._asstBuf = new Uint8Array(an.frequencyBinCount);
        }

        attachWebSocket(ws) {
            this._detachWs();
            this._ws = ws;
            this._wsHandler = (e) => this._onWsMessage(e);
            ws.addEventListener('message', this._wsHandler);
        }

        // One-shot: connect to a vui server. Grabs the mic, opens WS, sets up
        // WebRTC for the assistant audio, and binds everything to this orb.
        async connect(serverUrl) {
            // serverUrl: 'http(s)://host:port' or 'ws(s)://host:port' or '' (same origin)
            const base = (serverUrl || (location.origin)).replace(/\/$/, '');
            const httpBase = base.replace(/^ws/, 'http');
            const wsUrl = base.replace(/^http/, 'ws') + '/ws';

            // 1. mic
            const mic = await navigator.mediaDevices.getUserMedia({
                audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, sampleRate: 48000 },
            });
            this.attachMicStream(mic);

            // 2. WebRTC for assistant audio
            const pc = new RTCPeerConnection({ iceServers: [] });
            this._pc = pc;
            mic.getAudioTracks().forEach(t => pc.addTrack(t, mic));
            pc.ontrack = (e) => {
                const stream = e.streams[0];
                this.attachAssistantStream(stream);
                if (!this._remoteAudio) {
                    const a = document.createElement('audio');
                    a.autoplay = true;
                    a.style.display = 'none';
                    document.body.appendChild(a);
                    this._remoteAudio = a;
                }
                this._remoteAudio.srcObject = stream;
                this._remoteAudio.play().catch(() => {});
            };
            const offer = await pc.createOffer();
            await pc.setLocalDescription(offer);
            const resp = await fetch(httpBase + '/offer', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sdp: offer.sdp, type: offer.type }),
            });
            const answer = await resp.json();
            await pc.setRemoteDescription(new RTCSessionDescription(answer));

            // 3. WS for state events
            const ws = new WebSocket(wsUrl);
            this._ownsWs = true;
            ws.addEventListener('open', () => {
                ws.send(JSON.stringify({ type: 'vad_mode', enabled: true }));
            });
            ws.addEventListener('close', () => {
                if (this._destroyed) return;
                this._reconnectTimer = setTimeout(() => this.connect(serverUrl), 2000);
            });
            this.attachWebSocket(ws);
        }

        destroy() {
            this._destroyed = true;
            if (this._reconnectTimer) clearTimeout(this._reconnectTimer);
            this._detachWs();
            this._detachMic();
            this._detachAssistant();
            if (this._pc) { try { this._pc.close(); } catch (e) {} this._pc = null; }
            if (this._remoteAudio) { this._remoteAudio.remove(); this._remoteAudio = null; }
            if (this._ownsWs && this._ws) { try { this._ws.close(); } catch (e) {} }
            if (this.canvas && this.canvas.parentNode) this.canvas.parentNode.removeChild(this.canvas);
        }

        // ---------- internals ----------

        _detachMic() {
            if (this._micCtx) { try { this._micCtx.close(); } catch (e) {} }
            this._micCtx = null; this._micAnalyser = null; this._micBuf = null;
        }
        _detachAssistant() {
            if (this._asstCtx) { try { this._asstCtx.close(); } catch (e) {} }
            this._asstCtx = null; this._asstAnalyser = null; this._asstBuf = null;
        }
        _detachWs() {
            if (this._ws && this._wsHandler) this._ws.removeEventListener('message', this._wsHandler);
            this._wsHandler = null;
            this._ws = null;
        }

        _onWsMessage(e) {
            let data;
            try { data = JSON.parse(e.data); } catch { return; }
            switch (data.type) {
                case 'vad_start':
                    this.setState('listening');
                    break;
                case 'vad_stop':
                    this.setState('thinking');
                    break;
                case 'transcription':
                    this.setState('thinking');
                    break;
                case 'reply':
                case 'generating':
                    this.setState('speaking');
                    break;
                case 'turn_done':
                    this.setState('idle');
                    break;
            }
        }

        _buildDOM() {
            const c = document.createElement('canvas');
            c.style.display = 'block';
            c.style.width = '100%';
            c.style.height = '100%';
            this.canvas = c;
            this.ctx = c.getContext('2d');
            this.container.appendChild(c);
            this._resize();
            this._ro = new ResizeObserver(() => this._resize());
            this._ro.observe(this.container);
        }

        _resize() {
            const dpr = Math.min(window.devicePixelRatio || 1, 2);
            const rect = this.container.getBoundingClientRect();
            const w = this.opts.size || rect.width || 240;
            const h = this.opts.size || rect.height || 240;
            this.canvas.width = w * dpr;
            this.canvas.height = h * dpr;
            this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            this._w = w; this._h = h;
        }

        _readLevel(analyser, buf) {
            if (!analyser) return 0;
            analyser.getByteFrequencyData(buf);
            // Weighted: low-mid for "voice presence"
            const n = buf.length;
            const lo = Math.floor(n * 0.02);
            const hi = Math.floor(n * 0.4);
            let sum = 0;
            for (let i = lo; i < hi; i++) sum += buf[i];
            const avg = sum / (hi - lo) / 255;
            return clamp01(avg * 1.6);
        }

        _autoState() {
            // If WS hasn't told us otherwise, infer from levels.
            // Only auto-transition out of 'idle' / between listening/speaking
            // — leave 'thinking' alone (it's set by an explicit event).
            if (this._state === 'thinking') return;
            const m = this._micLevel, a = this._asstLevel;
            const speaking = a > 0.06 && a > m * 0.9;
            const listening = !speaking && m > 0.05;
            if (speaking) this.setState('speaking');
            else if (listening) this.setState('listening');
            else this.setState('idle');
        }

        _loop(now) {
            if (this._destroyed) return;
            requestAnimationFrame(this._loop);

            // Read levels
            const m = this._readLevel(this._micAnalyser, this._micBuf);
            const a = this._readLevel(this._asstAnalyser, this._asstBuf);
            // Smooth (asymmetric: fast attack, slow release)
            this._micLevel  = m > this._micLevel  ? lerp(this._micLevel,  m, 0.5) : lerp(this._micLevel,  m, 0.1);
            this._asstLevel = a > this._asstLevel ? lerp(this._asstLevel, a, 0.5) : lerp(this._asstLevel, a, 0.1);

            // If no WS is driving state, infer from audio
            if (!this._ws) this._autoState();

            // Theme lerp
            for (const k of ['hue', 'sat', 'light']) {
                // hue: take shortest path around the wheel
                if (k === 'hue') {
                    let d = this._targetTheme.hue - this._curTheme.hue;
                    if (d > 180) d -= 360; else if (d < -180) d += 360;
                    this._curTheme.hue = (this._curTheme.hue + d * 0.08 + 360) % 360;
                } else {
                    this._curTheme[k] = lerp(this._curTheme[k], this._targetTheme[k], 0.08);
                }
            }

            // Drive level (which audio source is "active" right now)
            const stateDrive =
                this._state === 'speaking' ? this._asstLevel
                : this._state === 'listening' ? this._micLevel
                : Math.max(this._micLevel, this._asstLevel) * 0.5;
            this._driveLevel = lerp(this._driveLevel, stateDrive, 0.18);

            // Idle breath
            this._breath = (now - this._t0) / 1000;

            this._render();
        }

        _render() {
            const ctx = this.ctx;
            const W = this._w, H = this._h;
            const cx = W / 2, cy = H / 2;
            const R = Math.min(W, H) * 0.32;

            // Clear
            ctx.globalCompositeOperation = 'source-over';
            if (this.opts.background === 'transparent') {
                ctx.clearRect(0, 0, W, H);
            } else {
                ctx.fillStyle = this.opts.background;
                ctx.fillRect(0, 0, W, H);
            }

            const breathing = (this._state === 'idle')
                ? 0.5 + 0.5 * Math.sin(this._breath * 1.4)
                : 0;
            const drive = clamp01(this._driveLevel + breathing * 0.18);

            const { hue, sat, light } = this._curTheme;

            // Soft outer glow
            if (this.opts.glow) {
                const glowR = R * (1.6 + drive * 0.6);
                const grad = ctx.createRadialGradient(cx, cy, R * 0.4, cx, cy, glowR);
                grad.addColorStop(0, `hsla(${hue}, ${sat}%, ${light}%, ${0.35 + drive * 0.35})`);
                grad.addColorStop(1, `hsla(${hue}, ${sat}%, ${light}%, 0)`);
                ctx.fillStyle = grad;
                ctx.beginPath();
                ctx.arc(cx, cy, glowR, 0, Math.PI * 2);
                ctx.fill();
            }

            // Layered morphing rings
            ctx.globalCompositeOperation = 'lighter';
            const N = this.opts.rings;
            const t = this._breath;
            for (let i = 0; i < N; i++) {
                const phase = t * (0.6 + i * 0.4) + i * 1.7;
                const ringR = R * (0.85 + i * 0.07);
                const layerDrive = drive * (1 - i * 0.18);
                const lightI = light + i * 5;
                const alpha = 0.55 - i * 0.13;
                const lineW = 2.5 + (N - i) * 0.6;

                ctx.beginPath();
                const steps = 96;
                for (let s = 0; s <= steps; s++) {
                    const th = (s / steps) * Math.PI * 2;
                    // Smooth pseudo-noise via summed sines (cheap, no deps)
                    const wob =
                        Math.sin(th * 3 + phase * 1.1) * 0.45 +
                        Math.sin(th * 5 - phase * 0.8) * 0.25 +
                        Math.sin(th * 2 + phase * 1.7 + i) * 0.30;
                    const r = ringR * (1 + wob * (0.06 + layerDrive * 0.32));
                    const x = cx + Math.cos(th) * r;
                    const y = cy + Math.sin(th) * r;
                    if (s === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
                }
                ctx.closePath();
                ctx.strokeStyle = `hsla(${hue + i * 6}, ${sat}%, ${lightI}%, ${alpha})`;
                ctx.lineWidth = lineW;
                ctx.stroke();
            }

            // Inner core
            ctx.globalCompositeOperation = 'source-over';
            const coreR = R * (0.55 + drive * 0.18);
            const core = ctx.createRadialGradient(cx, cy - coreR * 0.2, coreR * 0.1, cx, cy, coreR);
            core.addColorStop(0, `hsla(${hue}, ${sat}%, ${Math.min(95, light + 25)}%, ${0.85})`);
            core.addColorStop(0.6, `hsla(${hue}, ${sat}%, ${light}%, ${0.55})`);
            core.addColorStop(1, `hsla(${hue}, ${sat}%, ${Math.max(20, light - 25)}%, 0)`);
            ctx.fillStyle = core;
            ctx.beginPath();
            ctx.arc(cx, cy, coreR, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    // Web component wrapper
    if (typeof customElements !== 'undefined' && !customElements.get('vui-voice-viz')) {
        class VuiVoiceVizEl extends HTMLElement {
            connectedCallback() {
                if (this._viz) return;
                this.style.display = this.style.display || 'block';
                if (!this.style.width) this.style.width = '240px';
                if (!this.style.height) this.style.height = '240px';
                this._viz = new VuiVoiceVisualizer({ container: this });
                const src = this.getAttribute('src');
                if (src !== null) this._viz.connect(src);
            }
            disconnectedCallback() {
                if (this._viz) this._viz.destroy();
                this._viz = null;
            }
            get viz() { return this._viz; }
        }
        customElements.define('vui-voice-viz', VuiVoiceVizEl);
    }

    return VuiVoiceVisualizer;
}));
