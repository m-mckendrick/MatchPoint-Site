import pandas as pd
import numpy as np
from src.feature_engineering.match_context import add_pre_match_context_features, add_post_match_context_features


#black-box tests
#test output of functions

def test_basic_feature_outputs(p1p2_match_df):

    df=p1p2_match_df.copy()

    pre_features = add_pre_match_context_features(df)
    out = add_post_match_context_features(pre_features)

    #test outputs == num ect

    #rank features
    assert out["rank_diff"].iloc[0] == df["p2_rank"].iloc[0] - df["p1_rank"].iloc[0]
    assert out["rank_points_diff"].iloc[0] == df["p2_rank_points"].iloc[0] - df["p1_rank_points"].iloc[0]

    #handedness
    assert out["p1_lefty"].iloc[0] in (0, 1)
    assert out["p2_lefty"].iloc[0] in (0, 1)
    assert out["lefty_diff"].iloc[0] in (0, 1)

    #surface dummies
    assert any(col.startswith("surface_Hard")for col in out.columns)

    #round encoding eg(QF == 3)
    assert pd.notna(out["round_encoded"].iloc[0])

    assert out["match_year"].iloc[0] == df["tourney_date"].dt.year.iloc[0]

    assert out["age_diff"].iloc[0] == df["p2_age"].iloc[0] - df["p1_age"].iloc[0]
    assert out["height_diff"].iloc[0] == df["p2_ht"].iloc[0] - df["p1_ht"].iloc[0]

    assert out["tourney_prestige"].iloc[0] in (1,2,3,4)

    #test ret detection
    assert out["is_retirement"].iloc[0] in (0,1)