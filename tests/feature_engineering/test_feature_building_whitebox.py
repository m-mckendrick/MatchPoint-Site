import pandas as pd
from src.feature_engineering.feature_builder import build_features_from_df

def test_feature_builder_whitebox_minimal():
    #df with minimal data for features
    df = pd.DataFrame([{
        "p1_rank": 10,
        "p2_rank": 20,
        "p1_rank_points": 3000,
        "p2_rank_points": 1500,
        "p1_seed": 1,
        "p2_seed": 2,
        "p1_hand": "R",
        "p2_hand": "L",
        "surface": "Hard",
        "round": "QF",
        "draw_size": 32,
        "tourney_level": "A",
        "best_of": 3,
        "tourney_date": "2024-01-01",
        "p1_age": 30,
        "p2_age": 25,
        "p1_ht": 190,
        "p2_ht": 185,
        "score": "6-4 6-4",
        "p1_ace": 5,
        "p1_df": 1,
        "p1_svpt": 60,
        "p1_1stin": 40,
        "p1_1stwon": 30,
        "p1_2ndwon": 10,
        "p1_svgms": 10,
        "p1_bpsaved": 2,
        "p1_bpfaced": 2,
        "p2_ace": 3,
        "p2_df": 2,
        "p2_svpt": 55,
        "p2_1stin": 35,
        "p2_1stwon": 20,
        "p2_2ndwon": 8,
        "p2_svgms": 10,
        "p2_bpsaved": 1,
        "p2_bpfaced": 3,
    }])

    out = build_features_from_df(df.copy())

    #structure checks
    #returns df
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 1

    #check feature columns aren't in original df
    assert "rank_diff" not in df.columns
    assert "p1_first_serve_pct" not in df.columns

    #match context features
    expected_context_cols = [
        "rank_diff",
        "rank_points_diff",
        "p1_unseeded",
        "p2_unseeded",
        "seed_diff",
        "is_grand_slam",
        "is_masters",
        "is_best_of_five",
        "p1_lefty",
        "p2_lefty",
        "lefty_diff",
        "round_encoded",
        "match_year",
        "age_diff",
        "height_diff",
        "tourney_prestige",
    ]
    for col in expected_context_cols:
        assert col in out.columns

    #serve/return features
    expected_serve_cols = [
        "p1_first_serve_pct",
        "p1_first_serve_won_pct",
        "p1_second_serve_won_pct",
        "p1_df_rate",
        "p1_aces_per_game",
        "p1_bp_save_pct",
        "p1_spw_pct",
        "p1_rpw_pct",
        "p1_bp_conv_pct",
        "p1_sdi",
        "p2_first_serve_pct",
        "p2_df_rate",
    ]
    for col in expected_serve_cols:
        assert col in out.columns

    #check features are added with raw data
    preserved_cols = [
        "p1_rank",
        "p2_rank",
        "p1_rank_points",
        "p2_rank_points",
        "p1_seed",
        "p2_seed",
        "p1_hand",
        "p2_hand",
        "round",
        "draw_size",
        "tourney_level",
        "best_of",
        "tourney_date",
        "p1_age",
        "p2_age",
        "p1_ht",
        "p2_ht",
        "score",
        "p1_ace",
        "p1_df",
        "p1_svpt",
        "p1_1stin",
        "p1_1stwon",
        "p1_2ndwon",
        "p1_svgms",
        "p1_bpsaved",
        "p1_bpfaced",
        "p2_ace",
        "p2_df",
        "p2_svpt",
        "p2_1stin",
        "p2_1stwon",
        "p2_2ndwon",
        "p2_svgms",
        "p2_bpsaved",
        "p2_bpfaced",
    ]

    for col in preserved_cols:
        assert col in out.columns

    #surface get hot encoded to surface_{insert surface}
    assert "surface_Hard" in out.columns

    #no nas sanity check (make sure no override)
    assert pd.notna(out["rank_diff"].iloc[0])
    assert pd.notna(out["p1_first_serve_pct"].iloc[0])
