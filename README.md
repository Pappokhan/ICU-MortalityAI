# ICU-MortalityAI

A Streamlit app that estimates ICU/hospital mortality risk using a
**federated-learning XGBoost global ensemble**
(`FL_XGBoost_Global_Ensemble.pkl` — 5 client models, soft-voted). The
app is fully self-contained: it needs nothing besides this codebase and
the one `.pkl` file already included in `model/`.

## Architecture (MVC)

```
ICU-MortalityAI/
├── app.py                          # Entry point — orchestration only
├── config.py                       # Feature order, ranges, thresholds, example patients
├── model/                          # MODEL layer
│   ├── FL_XGBoost_Global_Ensemble.pkl
│   └── ml_model.py                 # Loads pkl, averages 5-model predictions, validates it on load
├── controller/                     # CONTROLLER layer
│   └── prediction_controller.py    # Form -> feature vector -> model call
├── view/                           # VIEW layer
│   ├── ui_components.py            # Streamlit rendering + HTML/CSS/JS glue
│   ├── report.py                   # Standalone downloadable HTML report generator
│   └── static/
│       ├── css/style.css           # Clinical dashboard theme
│       └── js/gauge.js             # Animated risk gauge (color zones + needle)
├── requirements.txt
├── Dockerfile
└── README.md
```

- **Model** (`model/ml_model.py`): loads the 5-model federated ensemble
  and validates it on startup (checks each object has `predict_proba`
  and expects the right number of features), raising a friendly
  `ModelLoadError` instead of crashing if the file is missing or the
  wrong shape. `FederatedXGBoostEnsemble.predict()` averages each
  client model's `predict_proba` output (FedAvg-style soft voting).
- **Controller** (`controller/prediction_controller.py`): takes the raw
  clinical values from the form, derives `age_risk_category` from age,
  min-max scales the continuous vitals, assembles them in the exact
  15-feature order the model expects, and calls the model.
- **View** (`view/ui_components.py`): renders the intake form, the
  live age-risk preview, an example-patient loader, the animated risk
  gauge (with color-coded risk zones and a needle), a plain-language
  interpretation of the result, and a "How this works" tab.

## What's new / fixed in this version

- **Downloadable HTML clinical report**: after an estimate, a "Download
  clinical report (.html)" button generates a standalone, print-ready
  HTML document (`view/report.py`) with the patient inputs, the result,
  per-model votes, the risk-tier table, and the disclaimer — shareable
  or printable without needing Streamlit.
- **Model agreement indicator**: the result card and report now show how
  closely the 5 federated clients agree with each other (1 − standard
  deviation of their votes), alongside the averaged probability.
- **Named feature vectors**: the controller now builds a `pandas`
  `DataFrame` with the exact training column names instead of a bare
  array, so predictions carry correct feature names into XGBoost.
- **Code cleanup**: all inline comments and docstrings were stripped
  from the Python/JS/CSS source for a leaner codebase; behavior is
  unchanged aside from the additions above.
- **Renamed** to ICU-MortalityAI throughout the UI, page title, and docs.
- **Live age-risk preview**: the age slider now lives outside the
  `st.form`, so the derived risk category updates immediately as you
  drag it (previously it only updated after clicking submit, because
  widgets inside a form don't rerun the page until submitted).
- **Example patients**: one-click "Stable post-op" / "Critically
  unstable" buttons pre-fill every field so first-time users can see a
  result without guessing 15 values, plus a "Reset to defaults" button.
- **Friendlier gauge**: color-coded risk zones (low/moderate/high/
  critical) with an animated needle, instead of a single plain arc —
  easier to read at a glance.
- **Plain-language result**: the result card now also says, in one
  sentence, what the percentage means ("out of 100 similar patients...").
- **"How this works" tab**: explains federated learning, the 15 input
  features, and how to read the risk tiers, in non-technical language.
- **Startup validation**: `model/ml_model.py` now checks the pickle
  exists, is unpickle-able, and matches the expected feature count
  before the app renders, showing a clear error card instead of a raw
  traceback if something's wrong.
- **Simplified scaling code**: removed a redundant "passthrough vs.
  scaled" special case in the controller — every continuous feature
  now goes through one min-max formula, configured per-feature in
  `config.FEATURE_META`.

## ⚠️ Accuracy note

The original notebook fit a `MinMaxScaler` on the full training set
before selecting the final 15 features, but that fitted scaler was
**not** exported alongside the model — only the 5 XGBoost boosters were
pickled. This app therefore rescales raw clinical inputs using
**clinically plausible min/max ranges** defined in `config.py`
(`FEATURE_META`), not the exact scaler statistics from training. This
is a close approximation, not an identical transform.

To make predictions reproduce the notebook's math exactly, re-run the
notebook's "Feature Scaling" cell and add:

```python
import joblib
joblib.dump(scaler, "minmax_scaler.pkl")
```

then place `minmax_scaler.pkl` in `model/` and load it in
`model/ml_model.py`, replacing the manual scaling in
`controller/prediction_controller.py::_min_max_scale`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run locally

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

## Deploy

**Streamlit Community Cloud**
1. Push this folder to a GitHub repo.
2. On share.streamlit.io, create a new app pointing at `app.py`.
3. It will auto-install from `requirements.txt`.

**Docker**
```bash
docker build -t icu-mortality-ai .
docker run -p 8501:8501 icu-mortality-ai
```

**Any VM / server**
```bash
pip install -r requirements.txt
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

## Disclaimer

This is a research/education demo built on a public Kaggle dataset. It
is **not** a certified medical device and must not be used for real
clinical decision-making.
