import pandas as pd
import numpy as np
from src.feature_engineering.form import add_form_features


def make_base_df():
    # 4 match dataset
    data = {
        "tourney_date": pd.to_datetime([
            "2024-01-01", "2024-01-05", "2024-01-10", "2024-01-15"
        ]),
        "match_num": [1, 2, 3, 4],
        "p1_id": [1, 1, 1, 1],
        "p2_id": [2, 3, 4, 5],
        "surface_raw": ["Hard", "Hard", "Clay", "Hard"],
        "label": [1, 0, 1, 0],  # win, loss, win, loss

        # minimal stats
        "p1_sets_won": [2, 1, 2, 0],
        "p1_sets_lost": [0, 2, 1, 2],
        "p1_games_won": [12, 10, 14, 6],
        "p1_games_lost": [6, 12, 10, 12],
        "p1_tiebreaks_won": [0, 0, 1, 0],
        "p2_tiebreaks_won": [0, 1, 0, 0],
        "p1_deciding_set_win": [0, 0, 1, 0],
        "p1_straight_sets_win": [1, 0, 0, 0],
        "p1_pressure_index": [0.5, 0.2, 0.8, 0.1],
        "p1_close_sets_won": [0, 0, 1, 0],

        # serve
        "p1_first_serve_pct": [0.60, 0.55, 0.62, 0.50],
        "p1_first_serve_won_pct": [0.70, 0.65, 0.72, 0.60],
        "p1_second_serve_won_pct": [0.50, 0.48, 0.55, 0.40],
        "p1_double_fault_rate": [0.03, 0.05, 0.02, 0.06],
        "p1_aces_per_game": [0.4, 0.3, 0.5, 0.2],
        "p1_serve_points_won_pct": [0.62, 0.58, 0.65, 0.55],
        "p1_estimated_hold_pct": [0.80, 0.75, 0.82, 0.70],
        "p1_serve_efficiency": [0.68, 0.63, 0.70, 0.58],

        # return
        "p1_return_points_won_pct": [0.38, 0.40, 0.42, 0.35],
        "p1_bp_conversion_pct": [0.30, 0.25, 0.35, 0.20],
        "p1_estimated_break_pct": [0.20, 0.18, 0.22, 0.15],
    }
    return pd.DataFrame(data)


def test_blackbox_form_all():
    df = make_base_df()
    out = add_form_features(df, windows=[2], surface_mode="all", match_selector=None, prefix_tag="all")

    # check that form columns exist
    expected_cols = [c for c in out.columns if "form_all_2" in c]
    assert len(expected_cols) > 0

    # first match must have all zeros
    first_row = out.iloc[0]
    for c in expected_cols:
        assert first_row[c] == 0

    # second match must use only match 1 stats
    second_row = out.iloc[1]
    assert second_row["p1_form_all_2_winrate"] == 1.0  # match 1 was a win


def test_blackbox_form_same():
    df = make_base_df()
    out = add_form_features(df, windows=[3], surface_mode="same", prefix_tag="same")

    # only match 3 is clay
    row3 = out.iloc[2]
    assert row3["p1_form_same_3_winrate"] == 0  # no past clay matches so 0


def test_blackbox_mirrored_df_not_double_counted():
    """
    Ensure that mirrored rows do NOT double-count matches.
    Player 1 plays one real match, but DF contains two mirrored rows.
    Form must treat it as ONE match.
    """

    df = pd.DataFrame({
        "tourney_date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
        "match_num": [1, 1],

        # mirrored rows
        "p1_id": [1, 2],
        "p2_id": [2, 1],

        "surface_raw": ["Hard", "Hard"],
        "label": [1, 0],

        # minimal stats
        "p1_sets_won": [2, 0],
        "p1_sets_lost": [0, 2],
        "p1_games_won": [12, 6],
        "p1_games_lost": [6, 12],
        "p1_tiebreaks_won": [0, 0],
        "p2_tiebreaks_won": [0, 0],
        "p1_deciding_set_win": [0, 0],
        "p1_straight_sets_win": [1, 0],
        "p1_pressure_index": [0.5, 0.1],
        "p1_close_sets_won": [0, 0],

        # serve
        "p1_first_serve_pct": [0.60, 0.50],
        "p1_first_serve_won_pct": [0.70, 0.60],
        "p1_second_serve_won_pct": [0.50, 0.40],
        "p1_double_fault_rate": [0.03, 0.06],
        "p1_aces_per_game": [0.4, 0.2],
        "p1_serve_points_won_pct": [0.62, 0.55],
        "p1_estimated_hold_pct": [0.80, 0.70],
        "p1_serve_efficiency": [0.68, 0.58],

        # return
        "p1_return_points_won_pct": [0.38, 0.35],
        "p1_bp_conversion_pct": [0.30, 0.20],
        "p1_estimated_break_pct": [0.20, 0.15],
    })

    out = add_form_features(df, windows=[1], surface_mode="all", match_selector=None , prefix_tag="all")

    # player 1 should see ONLY the row where p1_id == 1
    p1_row = out[out["p1_id"] == 1].iloc[0]

    # winrate must be 0 because no past matches exist
    assert p1_row["p1_form_all_1_winrate"] == 0

    # player 2 should see ONLY the row where p2_id == 2
    p2_row = out[out["p2_id"] == 2].iloc[0]
    assert p2_row["p2_form_all_1_winrate"] == 0

def test_no_future_leakage():
    df = make_base_df()
    out = add_form_features(df, windows=[3], surface_mode="all", match_selector=None , prefix_tag="all")
    # row 0 must always be zero
    assert out.iloc[0].filter(like="form_all_3").sum() == 0

