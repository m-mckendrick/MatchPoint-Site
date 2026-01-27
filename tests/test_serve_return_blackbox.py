import pandas as pd
import numpy as np
from src.feature_engineering.serve_return import add_serve_return_features


#black-box tests
#test output of functions

def test_serve_return_blackbox_single_match():
    #filler stats
    df = pd.DataFrame([{
        "w_ace": 8,
        "w_df": 2,
        "w_svpt": 74,
        "w_1stin": 52,
        "w_1stwon": 40,
        "w_2ndwon": 13,
        "w_svgms": 11,
        "w_bpsaved": 3,
        "w_bpfaced": 3,
        "l_ace": 9,
        "l_df": 3,
        "l_svpt": 95,
        "l_1stin": 58,
        "l_1stwon": 44,
        "l_2ndwon": 16,
        "l_svgms": 11,
        "l_bpsaved": 8,
        "l_bpfaced": 9,
    }])

    out = add_serve_return_features(df.copy())

    #winner expected values
    assert np.isclose(out["w_first_serve_pct"].iloc[0], 52/74)
    assert np.isclose(out["w_first_serve_won_pct"].iloc[0], 40/52)
    assert np.isclose(out["w_second_serve_won_pct"].iloc[0], 13/(74-52))
    assert np.isclose(out["w_df_rate"].iloc[0], 2/74)
    assert np.isclose(out["w_aces_per_game"].iloc[0], 8/11)
    assert np.isclose(out["w_bp_save_pct"].iloc[0], 3/3)
    assert np.isclose(out["w_spw_pct"].iloc[0], (40+13)/74)

    #return points won == 1 - opponents serve points won
    opp_spw = (44+16)/95
    assert np.isclose(out["w_rpw_pct"].iloc[0], 1 - opp_spw)

    #break point conversion == (opp_bpFaced - opp_bpSaved) / opp_bpFaced
    assert np.isclose(out["w_bp_conv_pct"].iloc[0], (9-8)/9)

    #serve dominance index = serve points won - return points won
    expected_sdi = ((40+13)/74) - (1 - opp_spw)
    assert np.isclose(out["w_sdi"].iloc[0], expected_sdi)

    #loser checks
    assert np.isclose(out["l_first_serve_pct"].iloc[0], 58/95)
    assert np.isclose(out["l_df_rate"].iloc[0], 3/95)
