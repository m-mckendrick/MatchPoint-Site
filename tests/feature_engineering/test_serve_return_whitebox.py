import pandas as pd
import numpy as np
from src.feature_engineering.serve_return import (
    compute_first_serve_pct,
    compute_first_serve_won_pct,
    compute_first_serve_in_play_won_pct,
    compute_second_serve_won_pct,
    compute_second_serve_in_play_won_pct,
    compute_double_fault_rate,
    compute_aces_per_game,
    compute_aces_per_first_serve_in,
    compute_first_serve_unreturned_pct,
    compute_bp_save_pct,
    compute_estimated_hold_pct,
    compute_serve_efficiency,
    compute_serve_points_won_pct,
    compute_return_points_won_pct,
    compute_bp_conversion_pct,
    compute_serve_dominance_index,
    compute_estimated_break_pct
)

#white-box tests
#these test every edge case and branch (every line of code)

#watching out for zero divison
def test_first_serve_pct_zero_svpt():
    df = pd.DataFrame({"1stin": [10], "svpt": [0]})
    assert np.isnan(compute_first_serve_pct(df).iloc[0])

def test_first_serve_won_pct_zero_first_in():
    df = pd.DataFrame({"1stwon": [10], "1stin": [0]})
    assert np.isnan(compute_first_serve_won_pct(df).iloc[0])

def test_first_serve_in_play_won_pct_zero_in_play():
    df = pd.DataFrame({"1stin": [10], "ace": [10], "df": [0], "1stwon": [10]})
    # all first serves were aces → no in‑play serves
    assert np.isnan(compute_first_serve_in_play_won_pct(df).iloc[0])

def test_first_serve_in_play_won_pct_basic():
    df = pd.DataFrame({"1stin": [20], "ace": [5], "df": [1], "1stwon": [12]})
    # in‑play = 20 - 5 - 1 = 14
    # in‑play won = 12 - 5 = 7
    assert np.isclose(compute_first_serve_in_play_won_pct(df).iloc[0], 7/14)


def test_second_serve_won_pct_zero_second_serve_points():
    df = pd.DataFrame({"2ndwon": [5], "svpt": [20], "1stin": [20]})
    assert np.isnan(compute_second_serve_won_pct(df).iloc[0])

def test_second_serve_in_play_won_pct_zero_in_play():
    df = pd.DataFrame({"svpt": [30], "1stin": [30], "df": [0], "2ndwon": [0]})
    # no second‑serve points at all
    assert np.isnan(compute_second_serve_in_play_won_pct(df).iloc[0])

def test_second_serve_in_play_won_pct_basic():
    df = pd.DataFrame({"svpt": [40], "1stin": [20], "df": [2], "2ndwon": [8]})
    # second serve pts = 20
    # in‑play = 20 - 2 = 18
    assert np.isclose(compute_second_serve_in_play_won_pct(df).iloc[0], 8/18)

def test_double_fault_rate_basic():
    df = pd.DataFrame({"df": [4], "svpt": [20]})
    assert compute_double_fault_rate(df).iloc[0] == 4/20

#cant have aces if you have no serve games
def test_aces_per_game_zero_games():
    df = pd.DataFrame({"ace": [5], "svgms": [0]})
    assert np.isnan(compute_aces_per_game(df).iloc[0])

def test_aces_per_first_serve_in_zero_first_in():
    df = pd.DataFrame({"ace": [5], "1stin": [0]})
    assert np.isnan(compute_aces_per_first_serve_in(df).iloc[0])

def test_aces_per_first_serve_in_basic():
    df = pd.DataFrame({"ace": [4], "1stin": [20]})
    assert np.isclose(compute_aces_per_first_serve_in(df).iloc[0], 4/20)

#cant have bp saved if not faced any
def test_bp_save_pct_zero_faced():
    df = pd.DataFrame({"bpsaved": [3], "bpfaced": [0]})
    assert np.isnan(compute_bp_save_pct(df).iloc[0])

def test_estimated_hold_pct_basic():
    df = pd.DataFrame({"1stwon": [30], "2ndwon": [10], "svpt": [50]})
    spw = (30+10)/50
    assert np.isclose(compute_estimated_hold_pct(df).iloc[0], spw**4)

def test_serve_efficiency_basic():
    df = pd.DataFrame({"1stwon": [30], "2ndwon": [10], "svpt": [50], "df": [5]})
    spw = (30+10)/50
    expected = spw - (5/50)
    assert np.isclose(compute_serve_efficiency(df).iloc[0], expected)

def test_estimated_break_pct_basic():
    df = pd.DataFrame({"dummy": [1]})  # unused
    opp = pd.DataFrame({"1stwon": [40], "2ndwon": [20], "svpt": [80]})
    opp_spw = (40+20)/80
    expected = 1 - (opp_spw**4)
    assert np.isclose(compute_estimated_break_pct(df, opp).iloc[0], expected)



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


def test_estimated_break_pct_basic():
    df = pd.DataFrame({"dummy": [1]})  # unused
    opp = pd.DataFrame({"1stwon": [40], "2ndwon": [20], "svpt": [80]})
    opp_spw = (40+20)/80
    expected = 1 - (opp_spw**4)
    assert np.isclose(compute_estimated_break_pct(df, opp).iloc[0], expected)
