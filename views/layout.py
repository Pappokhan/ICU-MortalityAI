"""VIEW layer — shared HTML/CSS helpers (template loading, CSS injection, header/footer)."""
import html
import re
from functools import lru_cache
from string import Template

import streamlit as st

import config


@lru_cache(maxsize=None)
def _template(name: str) -> Template:
    return Template((config.TEMPLATES_DIR / f"{name}.html").read_text(encoding="utf-8"))


def esc(value) -> str:
    """HTML-escape any dynamic text before it goes into a template."""
    return html.escape(str(value), quote=True)


def render_template(template_name: str, /, **values) -> str:
    return _template(template_name).substitute(values)


def show_html(markup: str) -> None:
    # Collapse to one line per tag block so Markdown never turns indented HTML into a code block.
    flat = " ".join(line.strip() for line in markup.splitlines() if line.strip())
    st.markdown(flat, unsafe_allow_html=True)


def inject_css() -> None:
    css = (config.ASSETS_DIR / "style.css").read_text(encoding="utf-8")
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    css = " ".join(line.strip() for line in css.splitlines() if line.strip())
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_header() -> None:
    show_html(render_template("header"))


def render_footer() -> None:
    show_html(render_template("footer"))


def render_error(title: str, message: str) -> None:
    show_html(render_template("error", title=esc(title), message=esc(message)))


def section(title: str, first: bool = False) -> None:
    show_html(render_template("section", title=esc(title), cls="icu-section--first" if first else ""))


def render_placeholder() -> None:
    show_html(render_template("placeholder"))
