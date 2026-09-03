import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "FL_XGBoost_Global_Ensemble.pkl")

APP_NAME = "ICU-MortalityAI"
APP_TAGLINE = "Federated XGBoost mortality risk estimation for ICU patients"
APP_ICON = "🩺"

FEATURE_ORDER = [
    "apache_4a_hospital_death_prob",
    "apache_4a_icu_death_prob",
    "gcs_verbal_apache",
    "gcs_eyes_apache",
    "gcs_motor_apache",
    "d1_spo2_min",
    "d1_sysbp_min",
    "d1_sysbp_noninvasive_min",
    "h1_spo2_max",
    "d1_mbp_min",
    "d1_mbp_noninvasive_min",
    "d1_temp_min",
    "d1_diasbp_min",
    "d1_diasbp_noninvasive_min",
    "age_risk_category",
]

DERIVED_FEATURES = {"age_risk_category"}

FEATURE_META = {
    "apache_4a_hospital_death_prob": {
        "label": "APACHE IVa — Predicted Hospital Death Probability",
        "help": "The APACHE IVa model's own predicted probability (0-1) of hospital death.",
        "min": 0.0, "max": 1.0, "default": 0.10, "step": 0.01,
        "scale_min": 0.0, "scale_max": 1.0, "unit": "",
    },
    "apache_4a_icu_death_prob": {
        "label": "APACHE IVa — Predicted ICU Death Probability",
        "help": "The APACHE IVa model's own predicted probability (0-1) of ICU death.",
        "min": 0.0, "max": 1.0, "default": 0.08, "step": 0.01,
        "scale_min": 0.0, "scale_max": 1.0, "unit": "",
    },
    "gcs_verbal_apache": {
        "label": "Glasgow Coma Scale — Verbal",
        "help": "Verbal response (1 = none, 5 = oriented & converses normally).",
        "min": 1, "max": 5, "default": 5, "step": 1,
        "scale_min": 1, "scale_max": 5, "unit": "",
    },
    "gcs_eyes_apache": {
        "label": "Glasgow Coma Scale — Eyes",
        "help": "Eye-opening response (1 = none, 4 = spontaneous).",
        "min": 1, "max": 4, "default": 4, "step": 1,
        "scale_min": 1, "scale_max": 4, "unit": "",
    },
    "gcs_motor_apache": {
        "label": "Glasgow Coma Scale — Motor",
        "help": "Motor response (1 = none, 6 = obeys commands).",
        "min": 1, "max": 6, "default": 6, "step": 1,
        "scale_min": 1, "scale_max": 6, "unit": "",
    },
    "d1_spo2_min": {
        "label": "Day-1 Minimum SpO₂",
        "help": "Lowest pulse-oximetry oxygen saturation recorded in the first 24h.",
        "min": 50, "max": 100, "default": 92, "step": 1,
        "scale_min": 0, "scale_max": 100, "unit": "%",
    },
    "d1_sysbp_min": {
        "label": "Day-1 Minimum Systolic BP",
        "help": "Lowest systolic blood pressure recorded in the first 24h.",
        "min": 40, "max": 250, "default": 95, "step": 1,
        "scale_min": 0, "scale_max": 250, "unit": "mmHg",
    },
    "d1_sysbp_noninvasive_min": {
        "label": "Day-1 Minimum Non-invasive Systolic BP",
        "help": "Lowest non-invasive (cuff) systolic BP in the first 24h.",
        "min": 40, "max": 250, "default": 95, "step": 1,
        "scale_min": 0, "scale_max": 250, "unit": "mmHg",
    },
    "h1_spo2_max": {
        "label": "Hour-1 Maximum SpO₂",
        "help": "Highest pulse-oximetry oxygen saturation in the first hour.",
        "min": 50, "max": 100, "default": 97, "step": 1,
        "scale_min": 0, "scale_max": 100, "unit": "%",
    },
    "d1_mbp_min": {
        "label": "Day-1 Minimum Mean Blood Pressure",
        "help": "Lowest mean arterial pressure recorded in the first 24h.",
        "min": 20, "max": 200, "default": 65, "step": 1,
        "scale_min": 0, "scale_max": 200, "unit": "mmHg",
    },
    "d1_mbp_noninvasive_min": {
        "label": "Day-1 Minimum Non-invasive Mean BP",
        "help": "Lowest non-invasive mean arterial pressure in the first 24h.",
        "min": 20, "max": 200, "default": 65, "step": 1,
        "scale_min": 0, "scale_max": 200, "unit": "mmHg",
    },
    "d1_temp_min": {
        "label": "Day-1 Minimum Temperature",
        "help": "Lowest core body temperature recorded in the first 24h.",
        "min": 30.0, "max": 42.0, "default": 36.0, "step": 0.1,
        "scale_min": 30.0, "scale_max": 42.0, "unit": "°C",
    },
    "d1_diasbp_min": {
        "label": "Day-1 Minimum Diastolic BP",
        "help": "Lowest diastolic blood pressure recorded in the first 24h.",
        "min": 20, "max": 150, "default": 55, "step": 1,
        "scale_min": 0, "scale_max": 150, "unit": "mmHg",
    },
    "d1_diasbp_noninvasive_min": {
        "label": "Day-1 Minimum Non-invasive Diastolic BP",
        "help": "Lowest non-invasive diastolic BP in the first 24h.",
        "min": 20, "max": 150, "default": 55, "step": 1,
        "scale_min": 0, "scale_max": 150, "unit": "mmHg",
    },
}

FORM_SECTIONS = [
    {
        "title": "Neurological status",
        "subtitle": "Glasgow Coma Scale — how responsive the patient is",
        "fields": ["gcs_verbal_apache", "gcs_eyes_apache", "gcs_motor_apache"],
    },
    {
        "title": "Oxygenation",
        "subtitle": "Blood oxygen levels in the first hours of admission",
        "fields": ["d1_spo2_min", "h1_spo2_max"],
    },
    {
        "title": "Blood pressure",
        "subtitle": "Lowest readings recorded in the first 24 hours",
        "fields": [
            "d1_sysbp_min", "d1_sysbp_noninvasive_min",
            "d1_mbp_min", "d1_mbp_noninvasive_min",
            "d1_diasbp_min", "d1_diasbp_noninvasive_min",
        ],
    },
    {
        "title": "Temperature",
        "subtitle": "Lowest core body temperature in the first 24 hours",
        "fields": ["d1_temp_min"],
    },
    {
        "title": "APACHE IVa severity scores",
        "subtitle": "The hospital's own severity-of-illness model output",
        "fields": ["apache_4a_hospital_death_prob", "apache_4a_icu_death_prob"],
    },
]


def age_to_risk_category(age: float) -> int:
    if age < 45:
        return 3
    elif 45 <= age <= 65:
        return 1
    elif 65 < age <= 80:
        return 0
    else:
        return 2


AGE_RISK_LABELS = {
    0: "Elderly — high baseline risk",
    1: "Middle-aged — moderate baseline risk",
    2: "Very elderly — critical baseline risk",
    3: "Young — low baseline risk",
}

RISK_THRESHOLDS = {
    "low": 0.20,
    "moderate": 0.45,
    "high": 0.70,
}

RISK_LEVEL_ORDER = ["low", "moderate", "high", "critical"]

EXAMPLE_PATIENTS = {
    "stable": {
        "label": "Stable post-op patient",
        "description": "Routine recovery, normal vitals and full alertness.",
        "values": {
            "age": 46,
            "apache_4a_hospital_death_prob": 0.03,
            "apache_4a_icu_death_prob": 0.02,
            "gcs_verbal_apache": 5,
            "gcs_eyes_apache": 4,
            "gcs_motor_apache": 6,
            "d1_spo2_min": 96,
            "d1_sysbp_min": 112,
            "d1_sysbp_noninvasive_min": 112,
            "h1_spo2_max": 99,
            "d1_mbp_min": 78,
            "d1_mbp_noninvasive_min": 78,
            "d1_temp_min": 36.6,
            "d1_diasbp_min": 70,
            "d1_diasbp_noninvasive_min": 70,
        },
    },
    "critical": {
        "label": "Critically unstable patient",
        "description": "Septic-shock-like picture: hypotensive, hypoxic, reduced consciousness.",
        "values": {
            "age": 78,
            "apache_4a_hospital_death_prob": 0.62,
            "apache_4a_icu_death_prob": 0.55,
            "gcs_verbal_apache": 2,
            "gcs_eyes_apache": 1,
            "gcs_motor_apache": 3,
            "d1_spo2_min": 82,
            "d1_sysbp_min": 74,
            "d1_sysbp_noninvasive_min": 76,
            "h1_spo2_max": 89,
            "d1_mbp_min": 48,
            "d1_mbp_noninvasive_min": 50,
            "d1_temp_min": 34.5,
            "d1_diasbp_min": 38,
            "d1_diasbp_noninvasive_min": 40,
        },
    },
}
