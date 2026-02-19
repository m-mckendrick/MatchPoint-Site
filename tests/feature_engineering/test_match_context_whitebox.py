import pandas as pd
import numpy as np
from src.feature_engineering.match_context import add_match_context_features, encode_round

#white-box tests
#these test every edge case and branch (every line of code)

def test_seed_diff_both_unseeded():
    df = pd.DataFrame([{
        "p1_rank": 10,
        "p2_rank": 20,
        "p1_rank_points": 3000,
        "p2_rank_points": 1500,
        "p1_seed": np.nan,
        "p2_seed": np.nan,
        "p1_hand": "R",
        "p2_hand": "R",
        "surface": "Clay",
        "round": "R16",
        "draw_size": 32,
        "tourney_level": "A",
        "best_of": 3,
        "tourney_date": "2024-01-01",
        "p1_age": 30,
        "p2_age": 25,
        "p1_ht": 190,
        "p2_ht": 185
    }])

    out = add_match_context_features(df)

    #both are unseeded so seed diff should be na
    assert np.isnan(out["seed_diff"].iloc[0])
    assert out["p1_unseeded"].iloc[0] == 1
    assert out["p2_unseeded"].iloc[0] == 1

#if round recorded as RR then encoded round cannot be encoded with algorithm
def test_encode_round_invalid_round():
    row = {
        "draw_size": 32,
        "round": "RR"
    }
    assert encode_round(row) is None

#test round encoder works if round number isn't a power of 2 (round is still 1st)
def test_encode_round_non_power_of_two_draw():
    row = {
        "draw_size": 28,
        "round": "R32"
    }
    assert encode_round(row) == 1

#if tourney level isn't known eg z (or could be one wasnt removed during cleaning like Olympics)
def test_prestige_unknown_level():
    df = pd.DataFrame([{
        "p1_rank": 10,
        "p2_rank": 20,
        "p1_rank_points": 3000,
        "p2_rank_points": 1500,
        "p1_seed": 1,
        "p2_seed": 2,
        "p1_hand": "R",
        "p2_hand": "R",
        "surface": "Grass",
        "round": "F",
        "draw_size": 32,
        "tourney_level": "Z",
        "best_of": 3,
        "tourney_date": "2024-01-01",
        "p1_age": 30,
        "p2_age": 25,
        "p1_ht": 190,
        "p2_ht": 185
    }])

    out = add_match_context_features(df)

    #prestige should be NaN
    assert np.isnan(out["tourney_prestige"].iloc[0])

#testing all the combination of the lefty matchup indicators
def test_lefty_logic_all_combinations():
    df = pd.DataFrame([
        {"p1_hand": "L", "p2_hand": "L"},
        {"p1_hand": "L", "p2_hand": "R"},
        {"p1_hand": "R", "p2_hand": "L"},
        {"p1_hand": "R", "p2_hand": "R"},
    ])

    df["p1_rank"] = 1
    df["p2_rank"] = 2
    df["p1_rank_points"] = 1000
    df["p2_rank_points"] = 900
    df["p1_seed"] = 1
    df["p2_seed"] = 2
    df["surface"] = "Hard"
    df["round"] = "R32"
    df["draw_size"] = 32
    df["tourney_level"] = "A"
    df["best_of"] = 3
    df["tourney_date"] = "2024-01-01"
    df["p1_age"] = 30
    df["p2_age"] = 25
    df["p1_ht"] = 190
    df["p2_ht"] = 185

    out = add_match_context_features(df)

    assert list(out["lefty_diff"]) == [0, 1, -1, 0]

def test_retirement_detection():
    df = pd.DataFrame([
        {"score": "6-1 6-1 1-0 RET"},
        {"score": "6-4,6-4"},
    ])

    df["p1_rank"] = 1
    df["p2_rank"] = 2
    df["p1_rank_points"] = 1000
    df["p2_rank_points"] = 900
    df["p1_seed"] = 1
    df["p2_seed"] = 2
    df["surface"] = "Hard"
    df["round"] = "R32"
    df["draw_size"] = 32
    df["tourney_level"] = "A"
    df["best_of"] = 3
    df["tourney_date"] = "2024-01-01"
    df["p1_age"] = 30
    df["p2_age"] = 25
    df["p1_ht"] = 190
    df["p2_ht"] = 185
    df["p1_hand"] = "L"
    df["p2_hand"] = "R"

    out = add_match_context_features(df)

    assert list(out["is_retirement"]) == [1,0]