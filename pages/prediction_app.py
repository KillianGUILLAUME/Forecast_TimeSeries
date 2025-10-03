from __future__ import annotations


from pathlib import Path
import sys


import streamlit as st





ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from streamlit_app_sections import init_session_state, render_prediction_page, show_header


def main() -> None:
    init_session_state()
    show_header()
    render_prediction_page()

if __name__ == "__main__":
    main()