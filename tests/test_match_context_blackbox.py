import pandas as pd
import numpy as np
from src.feature_engineering.match_context import add_match_context_features

#black-box tests
#test output of functions

def test_basic_feature_outputs():
    df = pd.DataFrame([{
        "winner_rank": 10,
        "loser_rank": 20,
        "winner_rank_points": 3000,
        "loser_rank_points": 1500,
        "winner_seed": 2,
        "loser_seed": 5,
        "winner_hand": "R",
        "loser_hand": "L",
        "surface": "Hard",
        "round": "QF",
        "draw_size": 32,
        "tourney_level": "A",
        "best_of": 3,
        "tourney_date": "2024-01-01",
        "winner_age": 30,
        "loser_age": 25,
        "winner_ht": 190,
        "loser_ht": 185
    }])

    out = add_match_context_features(df)

    #test outputs == num ect

    #rank features
    assert out["rank_diff"].iloc[0] == 10
    assert out["rank_points_diff"].iloc[0] == -1500

    #seed logic
    assert out["winner_unseeded"].iloc[0] == 0
    assert out["loser_unseeded"].iloc[0] == 0
    assert out["seed_diff"].iloc[0] == 3

    #handedness
    assert out["winner_lefty"].iloc[0] == 0
    assert out["loser_lefty"].iloc[0] == 1
    assert out["lefty_diff"].iloc[0] == -1

    #surface dummies
    assert out["surface_Hard"].iloc[0] == True

    #round encoding eg(QF == 3)
    assert out["round_encoded"].iloc[0] == 3

    assert out["match_year"].iloc[0] == 2024

    assert out["age_diff"].iloc[0] == -5
    assert out["height_diff"].iloc[0] == -5

    assert out["tourney_prestige"].iloc[0] == 2
