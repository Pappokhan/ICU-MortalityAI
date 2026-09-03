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
