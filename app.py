"""Entry point for the QuantIA Streamlit dashboard.

enter python -m streamlit run app.py in the terminal to start using the app
"""

from __future__ import annotations

import streamlit as st

from streamlit_app_sections import (
    init_session_state,
    render_home_page,
    show_header,
)


st.set_page_config(
    page_title="QuantIA – ETF & LSTM Analytics",
    page_icon="📈",
    layout="wide",
)


def main() -> None:
    init_session_state()
    show_header()
    render_home_page()

if __name__ == "__main__":
    main()