import pandas as pd
import numpy as np
from src.feature_engineering.match_context import add_match_context_features

#black-box tests
#test output of functions

def test_basic_feature_outputs(cleaned_match_df):

    df=cleaned_match_df.copy()

    out = add_match_context_features(df)

    #test outputs == num ect

    #rank features
    assert out["rank_diff"].iloc[0] == df["loser_rank"].iloc[0] - df["winner_rank"].iloc[0]
    assert out["rank_points_diff"].iloc[0] == df["loser_rank_points"].iloc[0] - df["winner_rank_points"].iloc[0]

    #seed logic
    assert out["winner_unseeded"].iloc[0] in (0,1)
    assert out["loser_unseeded"].iloc[0] in (0,1)
    if out["winner_unseeded"].iloc[0] == 0 and out["loser_unseeded"].iloc[0]==0:
        assert pd.notna(out["seed_diff"].iloc[0])
    else:
        assert pd.isna(out["seed_diff"].iloc[0])

    #handedness
    assert out["winner_lefty"].iloc[0] in (0, 1)
    assert out["loser_lefty"].iloc[0] in (0, 1)
    assert out["lefty_diff"].iloc[0] in (0, 1)

    #surface dummies
    assert any(col.startswith("surface_Hard")for col in out.columns)

    #round encoding eg(QF == 3)
    assert pd.notna(out["round_encoded"].iloc[0])

    assert out["match_year"].iloc[0] == df["tourney_date"].dt.year.iloc[0]

    assert out["age_diff"].iloc[0] == df["loser_age"].iloc[0] - df["winner_age"].iloc[0]
    assert out["height_diff"].iloc[0] == df["loser_ht"].iloc[0] - df["winner_ht"].iloc[0]

    assert out["tourney_prestige"].iloc[0] in (1,2,3,4)
