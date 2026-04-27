import pandas as pd
import numpy as np

from src.feature_engineering.serve_return import (
    compute_first_serve_pct,
    compute_first_serve_won_pct,
    compute_first_serve_in_play_won_pct,
    compute_second_serve_won_pct,
    compute_second_serve_in_play_won_pct,
    compute_double_fault_rate,
    compute_aces_per_game,
    compute_aces_per_first_serve_in,
    compute_first_serve_unreturned_pct,
    compute_bp_save_pct,
    compute_estimated_hold_pct,
    compute_serve_efficiency,
    compute_serve_points_won_pct,
    compute_return_points_won_pct,
    compute_bp_conversion_pct,
    compute_serve_dominance_index,
    compute_estimated_break_pct
)

#black-box tests
#test output of functions

def test_serve_return_blackbox_single_match(p1p2_match_df):

    df= p1p2_match_df.copy()

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

    #player only df
    p1 = pd.DataFrame({
        "svpt": [p1_svpt],
        "1stin": [p1_1stin],
        "1stwon": [p1_1stwon],
        "2ndwon": [p1_2ndwon],
        "df": [p1_df],
        "ace": [p1_ace],
        "svgms": [p1_svgms],
        "bpsaved": [p1_bpsaved],
        "bpfaced": [p1_bpfaced]
    })

    #opponent df
    p2 = pd.DataFrame({
        "svpt": [df["p2_svpt"].iloc[0]],
        "1stwon": [df["p2_1stwon"].iloc[0]],
        "2ndwon": [df["p2_2ndwon"].iloc[0]],
        "bpsaved": [df["p2_bpsaved"].iloc[0]],
        "bpfaced": [df["p2_bpfaced"].iloc[0]]
    })

    #serve stats
    assert np.isclose(compute_first_serve_pct(p1).iloc[0], p1_1stin / p1_svpt)
    assert np.isclose(compute_first_serve_won_pct(p1).iloc[0], p1_1stwon / p1_1stin)
    assert np.isclose(compute_second_serve_won_pct(p1).iloc[0], p1_2ndwon / (p1_svpt - p1_1stin))
    assert np.isclose(compute_double_fault_rate(p1).iloc[0], p1_df / p1_svpt)
    assert np.isclose(compute_aces_per_game(p1).iloc[0], p1_ace / p1_svgms)

    # first serve in-play win %
    p1_first_in_play = p1_1stin - p1_ace - p1_df
    if p1_first_in_play > 0:
        expected = (p1_1stwon - p1_ace) / p1_first_in_play
        assert np.isclose(compute_first_serve_in_play_won_pct(p1).iloc[0], expected)

    # second‑serve in‑play win %
    p1_second_pts = p1_svpt - p1_1stin
    p1_second_in_play = p1_second_pts - p1_df
    if p1_second_in_play > 0:
        expected = p1_2ndwon / p1_second_in_play
        assert np.isclose(compute_second_serve_in_play_won_pct(p1).iloc[0], expected)

    # aces per first serve in
    if p1_1stin > 0:
        assert np.isclose(compute_aces_per_first_serve_in(p1).iloc[0], p1_ace / p1_1stin)

    # first‑serve unreturned %
    if p1_1stin > 0:
        assert np.isclose(compute_first_serve_unreturned_pct(p1).iloc[0], p1_ace / p1_1stin)


    # break point save percentage
    if p1_bpfaced > 0:
        assert np.isclose(compute_bp_save_pct(p1).iloc[0], p1_bpsaved / p1_bpfaced)

    # serve points won percentage
    p1_spw = (p1_1stwon + p1_2ndwon) / p1_svpt
    assert np.isclose(compute_serve_points_won_pct(p1).iloc[0], p1_spw)

    # estimated hold %
    assert np.isclose(compute_estimated_hold_pct(p1).iloc[0], p1_spw ** 4)


    # serve efficiency
    expected_eff = p1_spw - (p1_df / p1_svpt)
    assert np.isclose(compute_serve_efficiency(p1).iloc[0], expected_eff)


    # return points won == 1 - opponents serve points won
    p2_1stwon = df["p2_1stwon"].iloc[0]
    p2_2ndwon = df["p2_2ndwon"].iloc[0]
    p2_svpt = df["p2_svpt"].iloc[0]
    opp_spw = (p2_1stwon + p2_2ndwon) / p2_svpt
    assert np.isclose(compute_return_points_won_pct(p1, p2).iloc[0], 1 - opp_spw)

    #break point conversion == (opp_bpFaced - opp_bpSaved) / opp_bpFaced
    p2_bpfaced = df["p2_bpfaced"].iloc[0]
    p2_bpsaved = df["p2_bpsaved"].iloc[0]
    if p2_bpfaced > 0:
        expected_conv = (p2_bpfaced - p2_bpsaved) / p2_bpfaced
        assert np.isclose(compute_bp_conversion_pct(p1, p2).iloc[0], expected_conv)

    #serve dominance index = serve points won - return points won
    expected_sdi = p1_spw - (1 - opp_spw)
    assert np.isclose(compute_serve_dominance_index(p1_spw, 1 - opp_spw), expected_sdi)

    #break percentage
    expected_break = 1 - (opp_spw ** 4)
    assert np.isclose(compute_estimated_break_pct(p1, p2).iloc[0], expected_break)

