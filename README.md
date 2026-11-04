# ICU-MortalityAI — Streamlit deployment (MVC)

Research prototype for the thesis. **Not a medical device; not for clinical use.**

## 1. Put the 9 artefacts in `artifacts/`

| File | From |
|---|---|
| `pipeline_artefacts.json`, `minmax_scaler.pkl`, `num_imputer.pkl` | Notebook 1 output |
| `tuned_xgb.pkl` | Notebook 2 output |
| `fl_client_1.pkl` … `fl_client_5.pkl` | Notebook 3 output |

(Files placed directly in the project root also work. `pipeline_artefacts.json` from Notebook 2 additionally
contains the tuned Youden threshold; the Notebook 1 copy works too.)

## 2. Install and run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Use the same `scikit-learn` (1.5.0) and a compatible `xgboost` version as the Kaggle notebooks, otherwise the
pickles may not load.

## Structure (MVC)

```
app.py                         entry point — wires M, V, C together
config.py                      paths and constants
models/                        M
  artifact_store.py              loads json + pkl artefacts (forces CPU)
  preprocessor.py                median impute -> MinMax scale (Notebook 1)
  predictor.py                   central XGBoost + 5-partition probability averaging
  risk.py                        Low / Moderate / High bands
  schema.py                      field labels, ranges, grouping
  report.py                      CaseReport data (inputs + prediction) for the PDF
views/                         V
  report_pdf.py                  builds the downloadable PDF report
  layout.py                      template loader, CSS injection, header/footer
  form_view.py                   input form (Streamlit widgets)
  result_view.py                 fills the HTML templates with a Prediction
  templates/*.html               HTML templates ($placeholders)
assets/style.css               all styling
controllers/                   C
  prediction_controller.py       form input -> raw features -> Prediction
artifacts/                     put the 9 notebook files here
```

## Notes
- **Layout:** one page, two columns — inputs on the left, results on the right.
- **Report:** after a prediction, click *Download report (PDF)* (inputs, risk band, partitions, drivers, disclaimer). Needs `reportlab`.
- The five "federated" models are partitions of one dataset, aggregated by probability averaging (not FedAvg).
- Probabilities come from SMOTE-balanced training and are not calibrated; read the risk bands.
