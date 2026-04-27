import pandas as pd
import numpy as np
from src.feature_engineering.form import (
    _extract_stats,
    _rolling_form_block,
    _form_keys,
    _compute_player_form
)


def test_extract_stats_whitebox():
    df = pd.DataFrame({
        "label": [1, 0],
        "p1_sets_won": [2, 1],
        "p1_sets_lost": [0, 2],
        "p1_games_won": [12, 10],
        "p1_games_lost": [6, 12],
        "p1_tiebreaks_won": [0, 1],
        "p2_tiebreaks_won": [0, 0],
        "p1_deciding_set_win": [0, 1],
        "p1_straight_sets_win": [1, 0],
        "p1_pressure_index": [0.5, 0.3],
        "p1_close_sets_won": [0, 1],
        "p1_first_serve_pct": [0.60, 0.55],
        "p1_first_serve_won_pct": [0.70, 0.65],
        "p1_second_serve_won_pct": [0.50, 0.48],
        "p1_double_fault_rate": [0.03, 0.05],
        "p1_aces_per_game": [0.4, 0.3],
        "p1_serve_points_won_pct": [0.62, 0.58],
        "p1_estimated_hold_pct": [0.80, 0.75],
        "p1_serve_efficiency": [0.68, 0.63],
        "p1_return_points_won_pct": [0.38, 0.40],
        "p1_bp_conversion_pct": [0.30, 0.25],
        "p1_estimated_break_pct": [0.20, 0.18],
        "p1_bp_save_pct": [0.40, 0.10],
    })

    stats = _extract_stats(df)

    assert "win" in stats
    assert stats["win"].tolist() == [1, 0]
    assert stats["first_serve_pct"].iloc[0] == 0.60


def test_rolling_form_block_whitebox():
    df = pd.DataFrame({
        "label": [1, 0, 1],
        "p1_sets_won": [2, 1, 2],
        "p1_sets_lost": [0, 2, 1],
        "p1_games_won": [12, 10, 14],
        "p1_games_lost": [6, 12, 10],
        "p1_tiebreaks_won": [0, 0, 1],
        "p2_tiebreaks_won": [0, 1, 0],
        "p1_deciding_set_win": [0, 0, 1],
        "p1_straight_sets_win": [1, 0, 0],
        "p1_pressure_index": [0.5, 0.2, 0.8],
        "p1_close_sets_won": [0, 0, 1],
        "p1_first_serve_pct": [0.60, 0.55, 0.62],
        "p1_first_serve_won_pct": [0.70, 0.65, 0.72],
        "p1_second_serve_won_pct": [0.50, 0.48, 0.55],
        "p1_double_fault_rate": [0.03, 0.05, 0.02],
        "p1_aces_per_game": [0.4, 0.3, 0.5],
        "p1_serve_points_won_pct": [0.62, 0.58, 0.65],
        "p1_estimated_hold_pct": [0.80, 0.75, 0.82],
        "p1_serve_efficiency": [0.68, 0.63, 0.70],
        "p1_return_points_won_pct": [0.38, 0.40, 0.42],
        "p1_bp_conversion_pct": [0.30, 0.25, 0.35],
        "p1_estimated_break_pct": [0.20, 0.18, 0.22],
        "p1_bp_save_pct": [0.40, 0.10, 0.35],
    })

    stats = _extract_stats(df)

    # Use last 2 matches
    past = df.iloc[:2]
    out = _rolling_form_block(past, stats, 2, "p1_", "all")

    assert out["p1_form_all_2_winrate"] == 0.5  # [1,0]
    assert out["p1_form_all_2_first_serve_pct"] == np.mean([0.60, 0.55])
    assert len(out) == len(_form_keys())


def test_rolling_form_block_empty_history_whitebox():
    df = pd.DataFrame({
        "label": [],
        "p1_sets_won": [],
        "p1_sets_lost": [],
        "p1_games_won": [],
        "p1_games_lost": [],
        "p1_tiebreaks_won": [],
        "p2_tiebreaks_won": [],
        "p1_deciding_set_win": [],
        "p1_straight_sets_win": [],
        "p1_pressure_index": [],
        "p1_close_sets_won": [],
        "p1_first_serve_pct": [],
        "p1_first_serve_won_pct": [],
        "p1_second_serve_won_pct": [],
        "p1_double_fault_rate": [],
        "p1_aces_per_game": [],
        "p1_serve_points_won_pct": [],
        "p1_estimated_hold_pct": [],
        "p1_serve_efficiency": [],
        "p1_return_points_won_pct": [],
        "p1_bp_conversion_pct": [],
        "p1_estimated_break_pct": [],
        "p1_bp_save_pct": [],
    })

    stats = _extract_stats(df)

    out = _rolling_form_block(df, stats, 5, "p1_", "all")

    # Should return zeros for all metrics
    assert all(v == 0 for v in out.values())

def test_compute_player_form_p2_uses_wrong_stats_whitebox():
    df = pd.DataFrame({
        "p1_id": [1],
        "p2_id": [2],
        "tourney_date": pd.to_datetime(["2024-01-01"]),
        "match_num": [1],
        "surface_raw": ["Hard"],
        "label": [1],
        "p1_sets_won": [2],
        "p1_sets_lost": [0],
        "p1_games_won": [12],
        "p1_games_lost": [6],
        "p1_tiebreaks_won": [0],
        "p2_tiebreaks_won": [0],
        "p1_deciding_set_win": [0],
        "p1_straight_sets_win": [1],
        "p1_pressure_index": [0.5],
        "p1_close_sets_won": [0],
        "p1_first_serve_pct": [0.60],
        "p1_first_serve_won_pct": [0.70],
        "p1_second_serve_won_pct": [0.50],
        "p1_double_fault_rate": [0.03],
        "p1_aces_per_game": [0.4],
        "p1_serve_points_won_pct": [0.62],
        "p1_estimated_hold_pct": [0.80],
        "p1_serve_efficiency": [0.68],
        "p1_return_points_won_pct": [0.38],
        "p1_bp_conversion_pct": [0.30],
        "p1_bp_save_pct": [0.10],
        "p1_estimated_break_pct": [0.20],
    })

    out = _compute_player_form(df.copy(), "p2_id", "p2_", [3], "all", None, "all")

    # p2 form should be zero because p2 has no stats
    # but your code incorrectly uses p1 stats
    p2_cols = [c for c in out.columns if c.startswith("p2_form_all_")]
    assert not all(out[c].iloc[0] != 0 for c in p2_cols), \
        "p2 form incorrectly computed using p1 stats"
