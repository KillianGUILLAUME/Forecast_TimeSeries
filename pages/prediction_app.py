from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from main import DEFAULT_LSTM_HP, run_lstm_training
from prediction_lstm_model import LSTMPredictorProba
from etf_collector import EuropeanETFCollector
from etf_visualizer import ETFVisualizer



ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from data_preprocessing import (
    build_lstm_features,
    prepare_lstm_training_datasets,
    resolve_training_universe,
)


st.title("Prediction and Training module")