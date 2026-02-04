import pandas as pd
from src.feature_engineering.feature_builder import build_features_from_df

def test_feature_builder_whitebox_minimal():
    #df with minimal data for features
    df = pd.DataFrame([{
        "winner_rank": 10,
        "loser_rank": 20,
        "winner_rank_points": 3000,
        "loser_rank_points": 1500,
        "winner_seed": 1,
        "loser_seed": 2,
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
        "loser_ht": 185,
        "score": "6-4 6-4",
        "w_ace": 5,
        "w_df": 1,
        "w_svpt": 60,
        "w_1stin": 40,
        "w_1stwon": 30,
        "w_2ndwon": 10,
        "w_svgms": 10,
        "w_bpsaved": 2,
        "w_bpfaced": 2,
        "l_ace": 3,
        "l_df": 2,
        "l_svpt": 55,
        "l_1stin": 35,
        "l_1stwon": 20,
        "l_2ndwon": 8,
        "l_svgms": 10,
        "l_bpsaved": 1,
        "l_bpfaced": 3,
    }])

    out = build_features_from_df(df.copy())

    #structure checks
    #returns df
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 1

    #check feature columns aren't in original df
    assert "rank_diff" not in df.columns
    assert "w_first_serve_pct" not in df.columns

    #match context features
    expected_context_cols = [
        "rank_diff",
        "rank_points_diff",
        "winner_unseeded",
        "loser_unseeded",
        "seed_diff",
        "is_grand_slam",
        "is_masters",
        "is_best_of_five",
        "winner_lefty",
        "loser_lefty",
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
        "w_first_serve_pct",
        "w_first_serve_won_pct",
        "w_second_serve_won_pct",
        "w_df_rate",
        "w_aces_per_game",
        "w_bp_save_pct",
        "w_spw_pct",
        "w_rpw_pct",
        "w_bp_conv_pct",
        "w_sdi",
        "l_first_serve_pct",
        "l_df_rate",
    ]
    for col in expected_serve_cols:
        assert col in out.columns

    #check features are added with raw data
    preserved_cols = [
        "winner_rank",
        "loser_rank",
        "winner_rank_points",
        "loser_rank_points",
        "winner_seed",
        "loser_seed",
        "winner_hand",
        "loser_hand",
        "round",
        "draw_size",
        "tourney_level",
        "best_of",
        "tourney_date",
        "winner_age",
        "loser_age",
        "winner_ht",
        "loser_ht",
        "score",
        "w_ace",
        "w_df",
        "w_svpt",
        "w_1stin",
        "w_1stwon",
        "w_2ndwon",
        "w_svgms",
        "w_bpsaved",
        "w_bpfaced",
        "l_ace",
        "l_df",
        "l_svpt",
        "l_1stin",
        "l_1stwon",
        "l_2ndwon",
        "l_svgms",
        "l_bpsaved",
        "l_bpfaced",
    ]

    for col in preserved_cols:
        assert col in out.columns

    #surface get hot encoded to surface_{insert surface}
    assert "surface_Hard" in out.columns

    #no nas sanity check (make sure no override)
    assert pd.notna(out["rank_diff"].iloc[0])
    assert pd.notna(out["w_first_serve_pct"].iloc[0])
