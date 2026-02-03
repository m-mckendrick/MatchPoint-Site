import pandas as pd
import numpy as np
from src.feature_engineering.serve_return import (
    compute_first_serve_pct,
    compute_first_serve_won_pct,
    compute_second_serve_won_pct,
    compute_double_fault_rate,
    compute_aces_per_game,
    compute_bp_save_pct,
    compute_serve_points_won_pct,
    compute_return_points_won_pct,
    compute_bp_conversion_pct,
    compute_serve_dominance_index
)

#white-box tests
#these test every edge case and branch (every line of code)

#watchng out for zero divison
def test_first_serve_pct_zero_svpt():
    df = pd.DataFrame({"1stin": [10], "svpt": [0]})
    assert np.isnan(compute_first_serve_pct(df).iloc[0])

def test_first_serve_won_pct_zero_first_in():
    df = pd.DataFrame({"1stwon": [10], "1stin": [0]})
    assert np.isnan(compute_first_serve_won_pct(df).iloc[0])

def test_second_serve_won_pct_zero_second_serve_points():
    df = pd.DataFrame({"2ndwon": [5], "svpt": [20], "1stin": [20]})
    assert np.isnan(compute_second_serve_won_pct(df).iloc[0])

def test_double_fault_rate_basic():
    df = pd.DataFrame({"df": [4], "svpt": [20]})
    assert compute_double_fault_rate(df).iloc[0] == 4/20

#cant have aces if you have no serve games
def test_aces_per_game_zero_games():
    df = pd.DataFrame({"ace": [5], "svgms": [0]})
    assert np.isnan(compute_aces_per_game(df).iloc[0])

#cant have bp saved if not faced any
def test_bp_save_pct_zero_faced():
    df = pd.DataFrame({"bpsaved": [3], "bpfaced": [0]})
    assert np.isnan(compute_bp_save_pct(df).iloc[0])

def test_return_points_won_pct_basic():
    p = pd.DataFrame({"1stwon": [30], "2ndwon": [10], "svpt": [60]})
    opp = pd.DataFrame({"1stwon": [40], "2ndwon": [20], "svpt": [80]})
    opp_spw = (40+20)/80
    assert np.isclose(compute_return_points_won_pct(p, opp).iloc[0], 1 - opp_spw)

def test_bp_conversion_pct_zero_faced():
    df = pd.DataFrame({"bpfaced": [0], "bpsaved": [0]})
    opp = pd.DataFrame({"bpfaced": [0], "bpsaved": [0]})
    assert np.isnan(compute_bp_conversion_pct(df, opp).iloc[0])

def test_sdi_basic():
    assert compute_serve_dominance_index(0.6, 0.3) == 0.3
