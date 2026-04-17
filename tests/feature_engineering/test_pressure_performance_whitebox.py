import pandas as pd
import numpy as np
from src.feature_engineering.pressure_performance import add_pressure_performance_features, safe_div


def test_safe_div():
    assert safe_div(4, 2) == 2
    assert safe_div(0, 5) == 0
    assert safe_div(5, 0) == 0
    assert safe_div(5, None) == 0


def test_deciding_set_best_of_3():
    df = pd.DataFrame([{
        "score": "6-4 3-6 6-2",
        "best_of": 3,
        "p1_bpsaved": 0, "p1_bpfaced": 1,
        "p2_bpsaved": 0, "p2_bpfaced": 1,
        "p1_tiebreaks_won": 0, "p2_tiebreaks_won": 0,
        "tourney_prestige": 1, "round_points": 20,
    }])

    out = add_pressure_performance_features(df.copy())
    assert out["p1_deciding_set_win"].iloc[0] == 1
    assert out["p2_deciding_set_win"].iloc[0] == 0


def test_deciding_set_best_of_5():
    df = pd.DataFrame([{
        "score": "6-4 3-6 6-3 4-6 6-2",
        "best_of": 5,
        "p1_bpsaved": 0, "p1_bpfaced": 1,
        "p2_bpsaved": 0, "p2_bpfaced": 1,
        "p1_tiebreaks_won": 0, "p2_tiebreaks_won": 0,
        "tourney_prestige": 1, "round_points": 20,
    }])

    out = add_pressure_performance_features(df.copy())
    assert out["p1_deciding_set_win"].iloc[0] == 1
    assert out["p2_deciding_set_win"].iloc[0] == 0


def test_close_set_logic():
    df = pd.DataFrame([{
        "score": "7-6(4) 6-4 6-2",
        "best_of": 3,
        "p1_bpsaved": 0, "p1_bpfaced": 1,
        "p2_bpsaved": 0, "p2_bpfaced": 1,
        "p1_tiebreaks_won": 1, "p2_tiebreaks_won": 0,
        "tourney_prestige": 1, "round_points": 20,
    }])

    out = add_pressure_performance_features(df.copy())

    assert out["p1_close_sets_won"].iloc[0] == 2   # 7-6, 6-4
    assert out["p2_close_sets_won"].iloc[0] == 0


def test_match_importance_and_multiplier():
    df = pd.DataFrame([
        {"score": "6-2 6-2", "best_of": 3,
         "p1_bpsaved": 0, "p1_bpfaced": 1,
         "p2_bpsaved": 0, "p2_bpfaced": 1,
         "p1_tiebreaks_won": 0, "p2_tiebreaks_won": 0,
         "tourney_prestige": 1, "round_points": 20},

        {"score": "6-2 6-2", "best_of": 3,
         "p1_bpsaved": 0, "p1_bpfaced": 1,
         "p2_bpsaved": 0, "p2_bpfaced": 1,
         "p1_tiebreaks_won": 0, "p2_tiebreaks_won": 0,
         "tourney_prestige": 4, "round_points": 1200},
    ])

    out = add_pressure_performance_features(df.copy())

    assert out.loc[0, "match_importance"] < out.loc[1, "match_importance"]
    assert out.loc[0, "pressure_multiplier"] < out.loc[1, "pressure_multiplier"]
