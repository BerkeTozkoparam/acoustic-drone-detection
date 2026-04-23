"""
Streamlit dashboard — real-time acoustic drone detection.

Local run  : streamlit run app.py   (live microphone + alarm sound)
Cloud run  : deployed on Streamlit Cloud (demo mode with synthetic audio)
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import librosa
import soundfile as sf
import subprocess

# Microphone + alarm sound only available locally
try:
    import sounddevice as sd
    MIC_AVAILABLE = True
except (ImportError, OSError):
    MIC_AVAILABLE = False

from acoustic_pipeline import (
    AcousticDronePipeline, DecisionEngine,
    SAMPLE_RATE, N_FFT, HOP_LENGTH, N_MFCC,
)

MODEL      = os.path.join(os.path.dirname(__file__), "drone_model.joblib")
_ALARM_WAV = os.path.join(os.path.dirname(__file__), ".alarm_tone.wav")

IS_CLOUD = not MIC_AVAILABLE  # running on Streamlit Cloud or similar


# ── Alarm sound (local only) ──────────────────────────────────
def _generate_alarm_wav(volume: float = 0.6):
    sr   = 44100
    dur  = 0.4
    t    = np.linspace(0, dur, int(sr * dur), endpoint=False)
    tone = (np.sin(2 * np.pi * 880  * t) * 0.5 +
            np.sin(2 * np.pi * 1320 * t) * 0.5) * volume
    fade = int(sr * 0.02)
    tone[-fade:] *= np.linspace(1, 0, fade)
    sf.write(_ALARM_WAV, tone.astype(np.float32), sr)


def play_alarm(volume: float = 0.6):
    """Play alarm via macOS afplay — avoids PortAudio stream conflicts."""
    if IS_CLOUD:
        return
    _generate_alarm_wav(volume)
    subprocess.Popen(["afplay", _ALARM_WAV])


# ── Synthetic audio for cloud demo ───────────────────────────
def _synthetic_frame(is_drone: bool = False) -> np.ndarray:
    """Generate a fake 1-second audio frame for demo purposes."""
    rng  = np.random.default_rng()
    freq = 120.0 if is_drone else 440.0
    t    = np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False)
    wave = np.sin(2 * np.pi * freq * t).astype(np.float32)
    wave += rng.normal(0, 0.05, SAMPLE_RATE).astype(np.float32)
    return wave


# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Drone Sound Detection",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@keyframes pulse { 0%{opacity:1} 50%{opacity:.6} 100%{opacity:1} }
.alarm {
    background:linear-gradient(135deg,#ff2222,#cc0000);
    color:#fff; padding:22px; border-radius:12px;
    text-align:center; font-size:1.8em; font-weight:700;
    animation:pulse 0.8s infinite; box-shadow:0 0 20px #ff4444;
}
.safe {
    background:linear-gradient(135deg,#00cc55,#009933);
    color:#fff; padding:22px; border-radius:12px;
    text-align:center; font-size:1.8em; font-weight:700;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────
def _init():
    if "pipeline" not in st.session_state:
        p = AcousticDronePipeline()
        try:
            p.classifier.load(MODEL)
        except FileNotFoundError:
            st.error(
                "**Model not found.**  \n"
                "Run `python acoustic_pipeline.py` locally to train and save "
                "`drone_model.joblib`, then redeploy."
            )
            st.stop()
        st.session_state.pipeline = p

    defaults = dict(
        running     = False,
        frame_count = 0,
        conf_hist   = [],
        alarm_hist  = [],
        log         = [],
        prev_alarm  = False,
        demo_drone  = False,   # cloud demo toggle
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    threshold   = st.slider("Drone confidence threshold", 0.1, 0.95, 0.60, 0.05,
                            help="Minimum probability to flag a frame as drone")
    window_size = st.slider("Sliding window size", 3, 15, 5,
                            help="Number of frames used for majority vote")
    alarm_ratio = st.slider("Alarm ratio", 0.2, 1.0, 0.60, 0.10,
                            help="Drone fraction in window required to trigger alarm")
    history_len = st.slider("Chart history length", 20, 200, 60)

    st.divider()
    st.subheader("🔔 Alarm Sound")
    if IS_CLOUD:
        st.caption("Sound unavailable on Streamlit Cloud.")
        sound_on = False
        volume   = 0.6
    else:
        sound_on = st.toggle("Enable alarm sound", value=True)
        volume   = st.slider("Volume", 0.1, 1.0, 0.6, 0.05, disabled=not sound_on)

    if IS_CLOUD:
        st.divider()
        st.subheader("🎭 Demo Mode")
        st.session_state.demo_drone = st.toggle(
            "Simulate drone audio",
            value=st.session_state.demo_drone,
            help="Feed synthetic drone-like audio instead of a real microphone"
        )

    st.divider()
    st.caption("Model: `drone_model.joblib`")
    st.caption(f"SR: {SAMPLE_RATE} Hz | Window: 1 s")
    st.caption(f"MFCC: {N_MFCC} | FFT: {N_FFT}")
    if IS_CLOUD:
        st.caption("⚠️ Running in cloud demo mode — no microphone access.")

    if st.button("🗑 Clear history", use_container_width=True):
        st.session_state.conf_hist  = []
        st.session_state.alarm_hist = []
        st.session_state.log        = []
        st.session_state.frame_count = 0
        st.rerun()


# ── Apply pipeline settings ───────────────────────────────────
pipeline = st.session_state.pipeline
pipeline.engine.alarm_ratio = alarm_ratio

if getattr(pipeline.engine, "_window_size", None) != window_size:
    pipeline.engine = DecisionEngine(window_size=window_size, alarm_ratio=alarm_ratio)
    pipeline.engine._window_size = window_size

import acoustic_pipeline as _ap
_ap.THRESHOLD = threshold


# ── Title ─────────────────────────────────────────────────────
st.title("🚁 Acoustic Drone Detection System")
mode_label = "☁️ Cloud demo mode — synthetic audio" if IS_CLOUD else "🎙️ Live microphone · MacBook Air"
st.caption(f"Real-time microphone-based drone detection · {mode_label}")

# ── Control buttons ───────────────────────────────────────────
c1, c2, _ = st.columns([1, 1, 5])
with c1:
    label    = "⏸ Stop"    if st.session_state.running else "▶ Start"
    btn_type = "secondary" if st.session_state.running else "primary"
    if st.button(label, type=btn_type, use_container_width=True):
        st.session_state.running = not st.session_state.running
        st.rerun()
with c2:
    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()

st.divider()

# ── Idle screen ───────────────────────────────────────────────
if not st.session_state.running:
    if IS_CLOUD:
        st.info(
            "☁️ **Cloud demo mode** — no microphone available.  \n"
            "Press **▶ Start** and toggle **Simulate drone audio** in the sidebar "
            "to see the dashboard in action.  \n"
            "For live detection, run `streamlit run app.py` locally."
        )
    else:
        st.info("Press **▶ Start**, then play a drone sound near the microphone.")

    if st.session_state.frame_count > 0:
        st.subheader("Last session summary")
        if st.session_state.conf_hist:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=st.session_state.conf_hist, mode="lines+markers",
                line=dict(color="#00ff88", width=2),
                marker=dict(color=["#ff4444" if a else "#00ff88"
                                   for a in st.session_state.alarm_hist], size=5),
                name="Confidence"
            ))
            fig.add_hline(y=threshold, line_dash="dash", line_color="yellow",
                          annotation_text="Threshold")
            fig.update_layout(
                height=250, margin=dict(l=0, r=0, t=10, b=0),
                yaxis=dict(range=[0, 1], title="Confidence", gridcolor="#333"),
                xaxis=dict(title="Frame", gridcolor="#333"),
                plot_bgcolor="#1a1a2e", paper_bgcolor="#1a1a2e", font_color="white"
            )
            st.plotly_chart(fig, use_container_width=True)
    st.stop()


# ── Capture / generate audio frame ───────────────────────────
if IS_CLOUD:
    audio = _synthetic_frame(is_drone=st.session_state.demo_drone)
    time.sleep(0.3)   # simulate ~1s capture on cloud
else:
    audio = pipeline.mic.capture()

decision    = pipeline.process_frame(audio)
conf        = decision["raw_conf"]
alarm       = decision["alarm"]
raw_label   = decision["raw_label"]
drone_ratio = decision["drone_ratio"]

# Play alarm only on the rising edge (False → True)
if sound_on and alarm and not st.session_state.prev_alarm:
    play_alarm(volume)
st.session_state.prev_alarm = alarm

# Update history
st.session_state.frame_count += 1
st.session_state.conf_hist.append(conf)
st.session_state.alarm_hist.append(int(alarm))
if len(st.session_state.conf_hist) > history_len:
    st.session_state.conf_hist  = st.session_state.conf_hist[-history_len:]
    st.session_state.alarm_hist = st.session_state.alarm_hist[-history_len:]

ts    = time.strftime("%H:%M:%S")
entry = (f"{ts}  {raw_label:<10}  conf={conf:.2f}  "
         f"ratio={drone_ratio:.2f}  {'🚨 ALARM' if alarm else '✅ Clear'}")
st.session_state.log.insert(0, entry)
st.session_state.log = st.session_state.log[:80]


# ── Status row ────────────────────────────────────────────────
s_col, m1, m2, m3 = st.columns([2, 1, 1, 1])

with s_col:
    if alarm:
        st.markdown('<div class="alarm">🚨 DRONE DETECTED</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="safe">✅ All Clear</div>',
                    unsafe_allow_html=True)

prev_conf = (st.session_state.conf_hist[-2]
             if len(st.session_state.conf_hist) > 1 else conf)
with m1:
    st.metric("Confidence", f"{conf*100:.1f}%",
              f"{(conf - prev_conf)*100:+.1f}%")
with m2:
    st.metric("Drone Ratio (window)", f"{drone_ratio*100:.0f}%")
with m3:
    st.metric("Frames Processed", st.session_state.frame_count)

st.divider()

# ── Middle row: waveform + confidence history ─────────────────
wc, cc = st.columns(2)

with wc:
    st.subheader("🔊 Live Waveform")
    t_axis = np.linspace(0, 1.0, len(audio))
    fig_w  = go.Figure()
    fig_w.add_trace(go.Scatter(
        x=t_axis, y=audio, mode="lines",
        line=dict(color="#00aaff" if not alarm else "#ff4444", width=1),
        name="PCM"
    ))
    fig_w.update_layout(
        height=220, margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(title="Time (s)", gridcolor="#333"),
        yaxis=dict(title="Amplitude", gridcolor="#333"),
        plot_bgcolor="#1a1a2e", paper_bgcolor="#1a1a2e", font_color="white"
    )
    st.plotly_chart(fig_w, use_container_width=True,
                    key=f"w{st.session_state.frame_count}")

with cc:
    st.subheader("📈 Confidence History")
    hist  = st.session_state.conf_hist
    ahist = st.session_state.alarm_hist
    fig_c = go.Figure()
    fig_c.add_hline(y=threshold, line_dash="dash", line_color="yellow",
                    annotation_text=f"Threshold {threshold:.2f}")
    fig_c.add_trace(go.Scatter(
        y=hist, mode="lines+markers",
        line=dict(color="#00ff88", width=2),
        marker=dict(color=["#ff4444" if a else "#00ff88" for a in ahist], size=5),
        name="Confidence"
    ))
    for i, a in enumerate(ahist):
        if a:
            fig_c.add_vrect(x0=i - 0.5, x1=i + 0.5,
                            fillcolor="#ff4444", opacity=0.15, line_width=0)
    fig_c.update_layout(
        height=220, margin=dict(l=0, r=0, t=0, b=0),
        yaxis=dict(range=[0, 1], title="Confidence", gridcolor="#333"),
        xaxis=dict(title="Frame #", gridcolor="#333"),
        plot_bgcolor="#1a1a2e", paper_bgcolor="#1a1a2e", font_color="white"
    )
    st.plotly_chart(fig_c, use_container_width=True,
                    key=f"c{st.session_state.frame_count}")

# ── Bottom row: MFCC heatmap + detection log ──────────────────
mc, lc = st.columns(2)

with mc:
    st.subheader("🎼 MFCC Spectrogram")
    mfcc = librosa.feature.mfcc(
        y=audio, sr=SAMPLE_RATE,
        n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    fig_m = go.Figure(data=go.Heatmap(
        z=mfcc, colorscale="Plasma",
        showscale=True, colorbar=dict(thickness=12)
    ))
    fig_m.update_layout(
        height=260, margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(title="Time frame"),
        yaxis=dict(title="MFCC coefficient"),
        plot_bgcolor="#1a1a2e", paper_bgcolor="#1a1a2e", font_color="white"
    )
    st.plotly_chart(fig_m, use_container_width=True,
                    key=f"m{st.session_state.frame_count}")

with lc:
    st.subheader("📋 Detection Log")
    st.code("\n".join(st.session_state.log[:18]) or "(no entries yet)",
            language=None)

# ── Auto-rerun ────────────────────────────────────────────────
time.sleep(0.05)
st.rerun()
