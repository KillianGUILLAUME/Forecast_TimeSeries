from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import requests


st.title("Graph Analysis module")