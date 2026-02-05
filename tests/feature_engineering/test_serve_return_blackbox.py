import pandas as pd
import numpy as np
from src.feature_engineering.serve_return import add_serve_return_features


#black-box tests
#test output of functions

def test_serve_return_blackbox_single_match(cleaned_match_df):

    df= cleaned_match_df.copy()

    out = add_serve_return_features(df)

    #winner expected values
    w_svpt = df["w_svpt"].iloc[0]
    w_1stin = df["w_1stin"].iloc[0]
    w_1stwon = df["w_1stwon"].iloc[0]
    w_2ndwon = df["w_2ndwon"].iloc[0]
    w_df = df["w_df"].iloc[0]
    w_ace = df["w_ace"].iloc[0]
    w_svgms = df["w_svgms"].iloc[0]
    w_bpsaved = df["w_bpsaved"].iloc[0]
    w_bpfaced = df["w_bpfaced"].iloc[0]

    #serve stats
    assert np.isclose(out["w_first_serve_pct"].iloc[0], w_1stin / w_svpt)
    assert np.isclose(out["w_first_serve_won_pct"].iloc[0], w_1stwon / w_1stin)
    assert np.isclose(out["w_second_serve_won_pct"].iloc[0], w_2ndwon / (w_svpt - w_1stin))
    assert np.isclose(out["w_df_rate"].iloc[0], w_df / w_svpt)
    assert np.isclose(out["w_aces_per_game"].iloc[0], w_ace / w_svgms)

    #break point save percentage
    if w_bpfaced > 0:
        assert np.isclose(out["w_bp_save_pct"].iloc[0], w_bpsaved / w_bpfaced)

    #serve points won percentage
    w_spw = (w_1stwon + w_2ndwon) / w_svpt
    assert np.isclose(out["w_spw_pct"].iloc[0], w_spw)

    #return points won == 1 - opponents serve points won
    l_1stwon = df["l_1stwon"].iloc[0]
    l_2ndwon = df["l_2ndwon"].iloc[0]
    l_svpt = df["l_svpt"].iloc[0]
    opp_spw = (l_1stwon + l_2ndwon) / l_svpt
    assert np.isclose(out["w_rpw_pct"].iloc[0], 1 - opp_spw)

    #break point conversion == (opp_bpFaced - opp_bpSaved) / opp_bpFaced
    l_bpfaced = df["l_bpfaced"].iloc[0]
    l_bpsaved = df["l_bpsaved"].iloc[0]
    if l_bpfaced > 0:
        assert np.isclose(out["w_bp_conv_pct"].iloc[0], (l_bpfaced - l_bpsaved) / l_bpfaced)

    #serve dominance index = serve points won - return points won
    expected_sdi = w_spw - (1 - opp_spw)
    assert np.isclose(out["w_sdi"].iloc[0], expected_sdi)

    #loser checks
    l_1stin = df["l_1stin"].iloc[0]
    assert np.isclose(out["l_first_serve_pct"].iloc[0], l_1stin / l_svpt)
    l_df = df["l_df"].iloc[0]
    assert np.isclose(out["l_df_rate"].iloc[0], l_df / l_svpt)
