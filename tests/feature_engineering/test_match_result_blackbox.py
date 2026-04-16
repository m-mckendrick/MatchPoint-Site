import pandas as pd
import numpy as np
from src.feature_engineering.match_results import add_match_result_features


def test_match_result_blackbox_basic_three_set_match():
    df = pd.DataFrame({
        "score": ["7-6(4) 6-7(8) 6-2"],
        "tiebreak_scores": ["7-4 10-8 none"],
        "p1_win": [1]
    })

    out = add_match_result_features(df.copy())

    # sets
    assert out["p1_sets_won"].iloc[0] == 2
    assert out["p2_sets_won"].iloc[0] == 1

    # games account for 7-6 mean 6 games each
    assert out["p1_games_won"].iloc[0] == 6 + 6 + 6
    assert out["p2_games_won"].iloc[0] == 6 + 6 + 2

    # tiebreaks
    assert out["p1_tiebreaks_won"].iloc[0] == 1
    assert out["p2_tiebreaks_won"].iloc[0] == 1

    # straight sets
    assert out["p1_straight_sets_win"].iloc[0] == 0
    assert out["p2_straight_sets_win"].iloc[0] == 0


def test_match_result_blackbox_straight_sets():
    df = pd.DataFrame({
        "score": ["6-4 6-4"],
        "tiebreak_scores": ["none none"],
        "p1_win": [1]
    })

    out = add_match_result_features(df.copy())

    assert out["p1_sets_won"].iloc[0] == 2
    assert out["p2_sets_won"].iloc[0] == 0
    assert out["p1_straight_sets_win"].iloc[0] == 1
    assert out["p2_straight_sets_win"].iloc[0] == 0


def test_match_result_blackbox_no_tiebreaks():
    df = pd.DataFrame({
        "score": ["6-3 3-6 6-4"],
        "tiebreak_scores": ["none none none"],
        "p1_win": [1]
    })

    out = add_match_result_features(df.copy())

    assert out["p1_tiebreaks_won"].iloc[0] == 0
    assert out["p2_tiebreaks_won"].iloc[0] == 0
