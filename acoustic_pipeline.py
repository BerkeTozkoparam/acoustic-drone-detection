"""
Acoustic Drone Detection Pipeline
==================================
Microphone → ADC (digitization) → Feature extraction → XGBoost / RandomForest classification

Setup:
    pip install sounddevice numpy librosa scikit-learn xgboost joblib datasets
"""

import numpy as np
import librosa

try:
    import sounddevice as sd
    SD_AVAILABLE = True
except (ImportError, OSError):
    sd = None
    SD_AVAILABLE = False
import joblib
import time
from collections import deque
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# ─────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────
SAMPLE_RATE   = 22050   # Hz — standard audio sample rate
DURATION      = 1.0     # seconds — length of each analysis window
N_MFCC        = 13      # number of MFCC coefficients
HOP_LENGTH    = 512
N_FFT         = 2048
THRESHOLD     = 0.6     # minimum drone probability to flag as drone
MODEL_PATH    = "drone_model.joblib"
CLASSES       = ["drone", "non_drone"]


# ─────────────────────────────────────────
# 2. STEP 1 — MICROPHONE / ADC LAYER
#    macOS CoreAudio → sounddevice → PCM
# ─────────────────────────────────────────
def find_mac_microphone() -> int | None:
    """
    Auto-detect the built-in microphone on macOS.
    Priority order:
      1. Input device whose name contains 'MacBook'
      2. Input device containing 'Built-in' or 'Microphone'
      3. System default input device
    Returns: device index (int) or None (use system default)
    """
    if not SD_AVAILABLE:
        return None

    assert sd is not None
    devices       = sd.query_devices()
    default_input = sd.default.device[0]

    candidates = []
    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] < 1:
            continue
        name = dev["name"].lower()
        if "macbook" in name:
            candidates.insert(0, (0, idx, dev["name"]))
        elif "built-in" in name or "microphone" in name or "built in" in name:
            candidates.append((1, idx, dev["name"]))
        else:
            candidates.append((2, idx, dev["name"]))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        _, idx, name = candidates[0]
        print(f"[Mic] Selected device → [{idx}] {name}")
        return idx

    print(f"[Mic] Auto-detect failed, using system default → [{default_input}]")
    return None


class MicrophoneCapture:
    """
    Real-time microphone capture via macOS CoreAudio.

    sounddevice uses PortAudio → CoreAudio under the hood.
    Works on Apple Silicon (M1/M2/M3) and Intel Macs.

    NOTE — macOS permissions:
      On first run the system will ask for microphone access.
      System Settings → Privacy & Security → Microphone → Terminal ✓

    When sounddevice is unavailable (e.g. Streamlit Cloud) all methods
    raise RuntimeError — use app.py's IS_CLOUD guard before calling capture().
    """

    def __init__(self, sample_rate=SAMPLE_RATE, duration=DURATION, device="auto"):
        self.sample_rate = sample_rate
        self.duration    = duration
        self.n_samples   = int(sample_rate * duration)

        if not SD_AVAILABLE:
            self.device = None
            return

        self.device = find_mac_microphone() if device == "auto" else device
        self._validate_device()

    def _validate_device(self):
        """Warn if the device's native sample rate differs from the target."""
        if not SD_AVAILABLE:
            return
        try:
            dev_info   = sd.query_devices(self.device, "input")
            default_sr = int(dev_info["default_samplerate"])
            if default_sr != self.sample_rate:
                print(
                    f"[Mic] Warning: device native SR={default_sr} Hz, "
                    f"target={self.sample_rate} Hz — librosa will resample."
                )
        except Exception:
            pass

    def capture(self) -> np.ndarray:
        """
        Record one analysis window.
        Returns: (n_samples,) float32 array.
        CoreAudio on macOS adds ~10–20 ms latency — expected behaviour.
        """
        if not SD_AVAILABLE:
            raise RuntimeError("sounddevice / PortAudio not available on this platform.")
        try:
            audio = sd.rec(
                frames     = self.n_samples,
                samplerate = self.sample_rate,
                channels   = 1,
                dtype      = "float32",
                device     = self.device,
                blocking   = True,
            )
            return audio.flatten()
        except sd.PortAudioError as e:
            print(f"[Mic] PortAudio error: {e}")
            print("      → System Settings > Privacy > Microphone > grant Terminal access")
            raise

    def list_devices(self):
        """Print all audio devices — use this to find the correct device index."""
        if not SD_AVAILABLE:
            print("[Mic] sounddevice not available — cannot list devices.")
            return
        print("\n── Audio Devices ──")
        for idx, dev in enumerate(sd.query_devices()):
            tag = ""
            if dev["max_input_channels"] > 0:
                tag += " [INPUT]"
            if dev["max_output_channels"] > 0:
                tag += " [OUTPUT]"
            print(f"  [{idx:2d}] {dev['name']:<40s} SR={int(dev['default_samplerate'])} Hz{tag}")
        inp, out = sd.default.device
        print(f"\n  Default input  → [{inp}]")
        print(f"  Default output → [{out}]\n")


# ─────────────────────────────────────────
# 3. STEP 2 — FEATURE EXTRACTION
#    Raw PCM → Acoustic fingerprint
# ─────────────────────────────────────────
class FeatureExtractor:
    """
    Extracts characteristic acoustic features of drone sounds:
      - MFCC               : Mel-frequency cepstral coefficients (timbre)
      - Spectral Centroid  : Frequency centre of mass
      - Spectral Rolloff   : Energy distribution threshold
      - RMS Energy         : Signal power
      - Zero Crossing Rate : Periodicity indicator
    """

    def __init__(self, sample_rate=SAMPLE_RATE, n_mfcc=N_MFCC):
        self.sample_rate = sample_rate
        self.n_mfcc      = n_mfcc

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        audio: (n_samples,) float32 array
        Returns: (feature_dim,) float64 feature vector
        """
        features = []

        # MFCC — 13 coefficients × (mean + std) = 26 features
        mfcc = librosa.feature.mfcc(
            y=audio, sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        features.extend(mfcc.mean(axis=1))
        features.extend(mfcc.std(axis=1))

        # Spectral Centroid (mean + std) = 2 features
        centroid = librosa.feature.spectral_centroid(
            y=audio, sr=self.sample_rate,
            n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        features.append(centroid.mean())
        features.append(centroid.std())

        # Spectral Rolloff = 2 features
        rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=self.sample_rate,
            n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        features.append(rolloff.mean())
        features.append(rolloff.std())

        # RMS Energy = 2 features
        rms = librosa.feature.rms(y=audio, hop_length=HOP_LENGTH)
        features.append(rms.mean())
        features.append(rms.std())

        # Zero Crossing Rate = 2 features
        zcr = librosa.feature.zero_crossing_rate(audio, hop_length=HOP_LENGTH)
        features.append(zcr.mean())
        features.append(zcr.std())

        return np.array(features)  # total: 26 + 2 + 2 + 2 + 2 = 34 features

    @property
    def feature_dim(self):
        return self.n_mfcc * 2 + 8   # 34


# ─────────────────────────────────────────
# 4. STEP 3 — CLASSIFIER
#    XGBoost (default) or RandomForest
# ─────────────────────────────────────────
class DroneClassifier:
    """
    Binary classifier: 0 = non_drone, 1 = drone.
    Supports training, saving, and loading.
    """

    def __init__(self, model_type="xgboost"):
        if model_type == "xgboost":
            self.model = XGBClassifier(
                n_estimators     = 200,
                max_depth        = 6,
                learning_rate    = 0.1,
                subsample        = 0.8,
                colsample_bytree = 0.8,
                eval_metric      = "logloss",
                random_state     = 42,
            )
        else:  # random_forest
            self.model = RandomForestClassifier(
                n_estimators = 300,
                max_depth    = None,
                n_jobs       = -1,
                random_state = 42,
            )
        self.model_type = model_type
        self.is_trained = False

    def train(self, X: np.ndarray, y: np.ndarray, test_size=0.2):
        """
        X: (n_samples, feature_dim)
        y: (n_samples,) — 0 or 1
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        print(f"[Model] Training: {self.model_type}")
        self.model.fit(X_train, y_train)
        self.is_trained = True

        y_pred = self.model.predict(X_test)
        print("\n── Test Results ──")
        print(classification_report(y_test, y_pred, labels=[0, 1], target_names=CLASSES))

    def predict(self, feature_vector: np.ndarray) -> dict:
        """
        Classify a single feature vector.
        Returns: { "label": str, "confidence": float, "is_drone": bool }
        """
        assert self.is_trained, "Model not trained — call train() or load() first."
        fv = feature_vector.reshape(1, -1)
        proba      = self.model.predict_proba(fv)[0]  # [p_non_drone, p_drone]
        drone_prob = proba[1]
        label      = "drone" if drone_prob >= THRESHOLD else "non_drone"
        return {
            "label"      : label,
            "confidence" : float(drone_prob),
            "is_drone"   : drone_prob >= THRESHOLD,
        }

    def save(self, path=MODEL_PATH):
        joblib.dump(self.model, path)
        print(f"[Model] Saved → {path}")

    def load(self, path=MODEL_PATH):
        self.model      = joblib.load(path)
        self.is_trained = True
        print(f"[Model] Loaded ← {path}")


# ─────────────────────────────────────────
# 5. STEP 4 — DECISION ENGINE
#    Smooths per-frame decisions over time
#    to suppress false alarms.
# ─────────────────────────────────────────
class DecisionEngine:
    """
    Sliding-window majority vote:
    Raises an alarm when the drone fraction across the last N
    frames exceeds alarm_ratio, filtering single-frame noise.
    """

    def __init__(self, window_size=5, alarm_ratio=0.6):
        self.window      = deque(maxlen=window_size)
        self.alarm_ratio = alarm_ratio

    def update(self, prediction: dict) -> dict:
        self.window.append(int(prediction["is_drone"]))
        drone_ratio = sum(self.window) / len(self.window)
        alarm       = drone_ratio >= self.alarm_ratio

        return {
            "raw_label"   : prediction["label"],
            "raw_conf"    : prediction["confidence"],
            "drone_ratio" : drone_ratio,
            "alarm"       : alarm,
            "status"      : "🚨 DRONE DETECTED" if alarm else "✅ Clear",
        }


# ─────────────────────────────────────────
# 6. FULL PIPELINE
# ─────────────────────────────────────────
class AcousticDronePipeline:
    """
    Top-level class that wires all layers together.

    Usage:
        pipeline = AcousticDronePipeline()
        pipeline.demo_train()       # train on synthetic data
        pipeline.run(n_frames=20)   # listen in real time for 20 windows
    """

    def __init__(self, model_type="xgboost"):
        self.mic        = MicrophoneCapture()
        self.extractor  = FeatureExtractor()
        self.classifier = DroneClassifier(model_type=model_type)
        self.engine     = DecisionEngine()

    def demo_train(self, n_samples=300):
        """Train on synthetic feature vectors when no real data is available."""
        print("[Demo] Generating synthetic training data...")
        feature_dim = self.extractor.feature_dim
        rng = np.random.default_rng(42)

        X_drone = rng.normal(loc=[0.6] * feature_dim, scale=0.15,
                             size=(n_samples // 2, feature_dim))
        X_non   = rng.normal(loc=[0.2] * feature_dim, scale=0.25,
                             size=(n_samples // 2, feature_dim))

        X = np.vstack([X_drone, X_non])
        y = np.array([1] * (n_samples // 2) + [0] * (n_samples // 2))

        self.classifier.train(X, y)

    def train_from_huggingface(self, split="train", max_samples=None):
        """
        Train on real audio from the HuggingFace DADS dataset.
        Dataset: geronimobasso/drone-audio-detection-samples
          Label 0 → non_drone
          Label 1 → drone

        Requires: pip install datasets
        The dataset is downloaded automatically on first run (~several GB).

        Args:
            split       : "train" or "test"
            max_samples : None → use all data, int → limit to first N samples
        """
        if not HF_AVAILABLE:
            raise ImportError(
                "HuggingFace datasets package not installed.\n"
                "Fix: pip install datasets"
            )

        print("[HF] Downloading dataset: geronimobasso/drone-audio-detection-samples")
        print("     (First run may take several minutes...)")

        ds = load_dataset(
            "geronimobasso/drone-audio-detection-samples",
            split=split,
            trust_remote_code=True,
        )

        ds = ds.shuffle(seed=42)
        if max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))

        print(f"[HF] {len(ds)} samples loaded. Extracting features...")

        X, y    = [], []
        skipped = 0

        for i, sample in enumerate(ds):
            try:
                audio   = np.array(sample["audio"]["array"], dtype=np.float32)
                orig_sr = sample["audio"]["sampling_rate"]

                if orig_sr != SAMPLE_RATE:
                    audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=SAMPLE_RATE)

                min_len = SAMPLE_RATE
                if len(audio) < min_len:
                    audio = np.pad(audio, (0, min_len - len(audio)))

                audio = audio[:SAMPLE_RATE]

                features = self.extractor.extract(audio)
                X.append(features)
                y.append(int(sample["label"]))

                if (i + 1) % 100 == 0:
                    print(f"  [{i+1}/{len(ds)}] processed...")

            except Exception as e:
                skipped += 1
                if skipped <= 5:
                    print(f"  [Warning] Sample {i} skipped: {e}")

        print(f"\n[HF] Done: {len(X)} samples processed, {skipped} skipped.")
        print(f"     Class distribution: drone={sum(y)}, non_drone={len(y)-sum(y)}")

        self.classifier.train(np.array(X), np.array(y))

    def process_frame(self, audio: np.ndarray) -> dict:
        """Run a single audio window through the full pipeline."""
        features   = self.extractor.extract(audio)
        prediction = self.classifier.predict(features)
        decision   = self.engine.update(prediction)
        return decision

    def run(self, n_frames=30):
        """
        Real-time inference loop.
        Captures and classifies one window per DURATION seconds.
        """
        if not self.classifier.is_trained:
            raise RuntimeError("Model not ready — call demo_train() or classifier.load() first.")

        print(f"\n[Pipeline] Listening — {n_frames} windows × {DURATION}s")
        print("─" * 50)

        for i in range(n_frames):
            audio    = self.mic.capture()
            decision = self.process_frame(audio)

            print(
                f"[{i+1:02d}] "
                f"Raw: {decision['raw_label']:10s} "
                f"({decision['raw_conf']:.2f}) | "
                f"Ratio: {decision['drone_ratio']:.2f} | "
                f"{decision['status']}"
            )

            if decision["alarm"]:
                self._trigger_alarm()

        print("─" * 50)
        print("[Pipeline] Done.")

    def _trigger_alarm(self):
        """Alarm action — extend this to send email, SMS, trigger camera, etc."""
        print("   ⚠️  Alarm triggered! (add notification logic here)")


# ─────────────────────────────────────────
# 7. ENTRY POINT
# ─────────────────────────────────────────
if __name__ == "__main__":
    pipeline = AcousticDronePipeline(model_type="xgboost")

    # ── TRAINING OPTIONS ─────────────────────────────────────────
    # Option A: Train on real HuggingFace data (recommended)
    #   Use max_samples=200 for a quick smoke-test.
    pipeline.train_from_huggingface(split="train", max_samples=None)

    # Option B: Synthetic data — fast, no download required
    # pipeline.demo_train(n_samples=500)
    # ─────────────────────────────────────────────────────────────

    # Save the trained model
    pipeline.classifier.save("drone_model.joblib")

    # On subsequent runs, skip training and just load:
    # pipeline.classifier.load("drone_model.joblib")

    # Real-time inference
    # pipeline.mic.list_devices()  # uncomment to list available audio devices
    pipeline.run(n_frames=20)
