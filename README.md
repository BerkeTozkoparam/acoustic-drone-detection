# Acoustic Drone Detection

Real-time drone detection from microphone audio using MFCC feature extraction and XGBoost classification, with a live Streamlit dashboard.

## Pipeline

```
Microphone → ADC → Feature Extraction → XGBoost → Decision Engine → Alert
```

**Feature vector (34 dimensions):**
- MFCC (13 coefficients × mean + std = 26)
- Spectral Centroid (mean + std = 2)
- Spectral Rolloff (mean + std = 2)
- RMS Energy (mean + std = 2)
- Zero Crossing Rate (mean + std = 2)

**Decision Engine:** Sliding-window majority vote over the last N frames to suppress single-frame false alarms.

## Setup

```bash
pip install -r requirements.txt
brew install libomp   # macOS — required by XGBoost
```

macOS microphone permission: **System Settings → Privacy & Security → Microphone → Terminal ✓**

## Train

```python
python acoustic_pipeline.py
```

Trains on the [geronimobasso/drone-audio-detection-samples](https://huggingface.co/datasets/geronimobasso/drone-audio-detection-samples) dataset from HuggingFace and saves `drone_model.joblib`.

## Run dashboard

```bash
streamlit run app.py
```

## Dashboard features

| Panel | Description |
|---|---|
| Status box | Red pulsing alarm / green clear indicator |
| Metrics | Live confidence score, drone ratio, frame count |
| Live Waveform | Real-time PCM plot (turns red on alarm) |
| Confidence History | Rolling chart with alarm regions highlighted |
| MFCC Spectrogram | Plasma heatmap updated each frame |
| Detection Log | Timestamped per-frame decision log |
| Alarm sound | 880 Hz + 1320 Hz tone via macOS `afplay` (toggle + volume in sidebar) |

## Results

Trained on 200 shuffled samples (100 drone / 100 non-drone):

```
              precision    recall  f1-score   support
       drone       1.00      1.00      1.00        37
   non_drone       1.00      1.00      1.00         3
    accuracy                           1.00        40
```

## Stack

- **Audio capture:** sounddevice (PortAudio → CoreAudio)
- **Feature extraction:** librosa
- **Classifier:** XGBoost (default) / RandomForest
- **Dashboard:** Streamlit + Plotly
- **Dataset:** HuggingFace `datasets`
