"""MODEL layer — description of the input fields (labels, ranges, grouping)."""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Field:
    key: str
    label: str
    kind: str = "float"            # "float" | "int" | "age" | "choice"
    min: Optional[float] = None
    max: Optional[float] = None
    default: Optional[float] = None
    step: Optional[float] = None


FIELDS = {
    "apache_4a_hospital_death_prob": Field("apache_4a_hospital_death_prob",
        "APACHE IV predicted hospital death probability", "float", 0.0, 1.0, 0.10, 0.01),
    "apache_4a_icu_death_prob": Field("apache_4a_icu_death_prob",
        "APACHE IV predicted ICU death probability", "float", 0.0, 1.0, 0.05, 0.01),
    "gcs_verbal_apache": Field("gcs_verbal_apache", "GCS verbal (1–5)", "int", 1, 5, 5, 1),
    "gcs_eyes_apache": Field("gcs_eyes_apache", "GCS eyes (1–4)", "int", 1, 4, 4, 1),
    "gcs_motor_apache": Field("gcs_motor_apache", "GCS motor (1–6)", "int", 1, 6, 6, 1),
    "h1_resprate_min": Field("h1_resprate_min", "Respiratory rate — min, first hour (/min)", "float", 0.0, 80.0, 14.0, 1.0),
    "h1_temp_min": Field("h1_temp_min", "Temperature — min, first hour (°C)", "float", 25.0, 43.0, 36.5, 0.1),
    "d1_sysbp_min": Field("d1_sysbp_min", "Systolic BP — min, first 24 h (mmHg)", "float", 20.0, 250.0, 100.0, 1.0),
    "d1_sysbp_noninvasive_min": Field("d1_sysbp_noninvasive_min",
        "Systolic BP non-invasive — min, first 24 h (mmHg)", "float", 20.0, 250.0, 100.0, 1.0),
    "d1_spo2_min": Field("d1_spo2_min", "SpO₂ — min, first 24 h (%)", "float", 0.0, 100.0, 94.0, 1.0),
    "h1_spo2_max": Field("h1_spo2_max", "SpO₂ — max, first hour (%)", "float", 0.0, 100.0, 99.0, 1.0),
    "d1_lactate_min": Field("d1_lactate_min", "Lactate — min, first 24 h (mmol/L)", "float", 0.0, 30.0, 1.5, 0.1),
    "d1_bilirubin_max": Field("d1_bilirubin_max", "Bilirubin — max, first 24 h (mg/dL)", "float", 0.0, 60.0, 0.8, 0.1),
    "age_risk_category": Field("age_risk_category", "Age (years)", "age", 16, 110, 65, 1),
    "ventilated_apache": Field("ventilated_apache", "Ventilated on ICU day 1?", "choice"),
}

# Section title -> feature keys (only keys that exist in the trained model are shown)
GROUPS = [
    ("Severity scores", ["apache_4a_hospital_death_prob", "apache_4a_icu_death_prob"]),
    ("Neurological (Glasgow Coma Scale)", ["gcs_verbal_apache", "gcs_eyes_apache", "gcs_motor_apache"]),
    ("Vital signs", ["h1_resprate_min", "h1_temp_min", "d1_sysbp_min",
                     "d1_sysbp_noninvasive_min", "d1_spo2_min", "h1_spo2_max"]),
    ("Laboratory", ["d1_lactate_min", "d1_bilirubin_max"]),
    ("Patient & support", ["age_risk_category", "ventilated_apache"]),
]

VENTILATED_CHOICES = ["Unknown", "No", "Yes"]


def label_for(key: str) -> str:
    f = FIELDS.get(key)
    if not f:
        return key
    return "Age risk category" if f.kind == "age" else f.label
