import pandas as pd
import numpy as np
from src.feature_engineering.serve_return import add_serve_return_features


#black-box tests
#test output of functions

def test_serve_return_blackbox_single_match(p1p2_match_df):

    df= p1p2_match_df.copy()

    out = add_serve_return_features(df)

    #winner expected values
    p1_svpt = df["p1_svpt"].iloc[0]
    p1_1stin = df["p1_1stin"].iloc[0]
    p1_1stwon = df["p1_1stwon"].iloc[0]
    p1_2ndwon = df["p1_2ndwon"].iloc[0]
    p1_df = df["p1_df"].iloc[0]
    p1_ace = df["p1_ace"].iloc[0]
    p1_svgms = df["p1_svgms"].iloc[0]
    p1_bpsaved = df["p1_bpsaved"].iloc[0]
    p1_bpfaced = df["p1_bpfaced"].iloc[0]

    #serve stats
    assert np.isclose(out["p1_first_serve_pct"].iloc[0], p1_1stin / p1_svpt)
    assert np.isclose(out["p1_first_serve_won_pct"].iloc[0], p1_1stwon / p1_1stin)
    assert np.isclose(out["p1_second_serve_won_pct"].iloc[0], p1_2ndwon / (p1_svpt - p1_1stin))
    assert np.isclose(out["p1_df_rate"].iloc[0], p1_df / p1_svpt)
    assert np.isclose(out["p1_aces_per_game"].iloc[0], p1_ace / p1_svgms)

    #break point save percentage
    if p1_bpfaced > 0:
        assert np.isclose(out["p1_bp_save_pct"].iloc[0], p1_bpsaved / p1_bpfaced)

    #serve points won percentage
    p1_spw = (p1_1stwon + p1_2ndwon) / p1_svpt
    assert np.isclose(out["p1_spw_pct"].iloc[0], p1_spw)

    #return points won == 1 - opponents serve points won
    p2_1stwon = df["p2_1stwon"].iloc[0]
    p2_2ndwon = df["p2_2ndwon"].iloc[0]
    p2_svpt = df["p2_svpt"].iloc[0]
    opp_spw = (p2_1stwon + p2_2ndwon) / p2_svpt
    assert np.isclose(out["p1_rpw_pct"].iloc[0], 1 - opp_spw)

    #break point conversion == (opp_bpFaced - opp_bpSaved) / opp_bpFaced
    p2_bpfaced = df["p2_bpfaced"].iloc[0]
    p2_bpsaved = df["p2_bpsaved"].iloc[0]
    if p2_bpfaced > 0:
        assert np.isclose(out["p1_bp_conv_pct"].iloc[0], (p2_bpfaced - p2_bpsaved) / p2_bpfaced)

    #serve dominance index = serve points won - return points won
    expected_sdi = p1_spw - (1 - opp_spw)
    assert np.isclose(out["p1_sdi"].iloc[0], expected_sdi)

    #loser checks
    p2_1stin = df["p2_1stin"].iloc[0]
    assert np.isclose(out["p2_first_serve_pct"].iloc[0], p2_1stin / p2_svpt)
    p2_df = df["p2_df"].iloc[0]
    assert np.isclose(out["p2_df_rate"].iloc[0], p2_df / p2_svpt)
