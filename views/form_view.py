"""VIEW layer — the patient input form. Returns raw widget values; no logic."""
from typing import Dict, List

import streamlit as st

from models.schema import FIELDS, GROUPS, VENTILATED_CHOICES, Field
from views.layout import section


def _number(field: Field, disabled: bool):
    key = f"in_{field.key}"
    if field.kind in ("int", "age"):
        return st.number_input(field.label, int(field.min), int(field.max),
                               int(field.default), int(field.step), key=key, disabled=disabled)
    return st.number_input(field.label, float(field.min), float(field.max),
                           float(field.default), float(field.step), key=key, disabled=disabled)


def _render_field(key: str):
    field = FIELDS.get(key) or Field(key, key, "float", -1e6, 1e6, 0.0, 1.0)

    if field.kind == "choice":
        return st.selectbox(field.label, VENTILATED_CHOICES, index=1, key=f"in_{key}")

    try:                                   # Streamlit >= 1.36: bottom-align checkbox with the input box
        box, na_col = st.columns([4, 1], vertical_alignment="bottom")
        spacer = False
    except TypeError:                      # older Streamlit: fall back to a CSS spacer
        box, na_col = st.columns([4, 1])
        spacer = True
    with na_col:
        if spacer:
            st.markdown('<div class="icu-na-spacer"></div>', unsafe_allow_html=True)
        na = st.checkbox("N/A", key=f"na_{key}",
                         help="Value unknown — replaced by the training-set median")
    with box:
        value = _number(field, disabled=na)
    return None if na else value


def render_form(features: List[str]) -> Dict[str, object]:
    st.caption("Tick **N/A** for any value that is not available; it is replaced by the training-set median.")
    values: Dict[str, object] = {}
    placed = set()

    for title, keys in GROUPS:
        present = [k for k in keys if k in features]
        if not present:
            continue
        section(title)
        cols = st.columns(2)
        for i, k in enumerate(present):
            with cols[i % 2]:
                values[k] = _render_field(k)
            placed.add(k)

    leftovers = [f for f in features if f not in placed]
    if leftovers:
        section("Other features")
        cols = st.columns(2)
        for i, k in enumerate(leftovers):
            with cols[i % 2]:
                values[k] = _render_field(k)
    return values
