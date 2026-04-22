import pandas as pd
from src.feature_engineering.form_profiles import (
    add_form_all,
    add_form_same,
    add_h2h_form_all,
    add_h2h_form_same,
    add_tournament_form_all

)


def make_df():
    """
    Create a small mirrored-safe dataset for testing form profiles.
    """
    data = {
        "tourney_date": pd.to_datetime([
            "2024-01-01", "2024-01-01",
            "2024-06-01", "2024-06-01",
            "2023-06-01", "2023-06-01",
        ]),
        "match_num": [1, 1, 2, 2, 3, 3],

        # mirrored rows
        "p1_id": [1, 2, 1, 3, 1, 4],
        "p2_id": [2, 1, 3, 1, 4, 1],

        "surface_raw": ["Hard", "Hard", "Clay", "Clay", "Hard", "Hard"],
        "label": [1, 0, 0, 1, 1, 0],

        # minimal stats
        "p1_sets_won": [2, 0, 1, 2, 2, 0],
        "p1_sets_lost": [0, 2, 2, 1, 0, 2],
        "p1_games_won": [12, 6, 10, 14, 12, 5],
        "p1_games_lost": [6, 12, 12, 10, 6, 12],
        "p1_tiebreaks_won": [0, 0, 0, 1, 0, 0],
        "p2_tiebreaks_won": [0, 0, 1, 0, 0, 0],
        "p1_deciding_set_win": [0, 0, 0, 1, 0, 0],
        "p1_straight_sets_win": [1, 0, 0, 0, 1, 0],
        "p1_pressure_index": [0.5, 0.1, 0.3, 0.7, 0.6, 0.2],
        "p1_close_sets_won": [0, 0, 0, 1, 0, 0],

        # serve
        "p1_first_serve_pct": [0.60] * 6,
        "p1_first_serve_won_pct": [0.70] * 6,
        "p1_second_serve_won_pct": [0.50] * 6,
        "p1_double_fault_rate": [0.03] * 6,
        "p1_aces_per_game": [0.4] * 6,
        "p1_serve_points_won_pct": [0.62] * 6,
        "p1_estimated_hold_pct": [0.80] * 6,
        "p1_serve_efficiency": [0.68] * 6,

        # return
        "p1_return_points_won_pct": [0.38] * 6,
        "p1_bp_conversion_pct": [0.30] * 6,
        "p1_estimated_break_pct": [0.20] * 6,

        # tournament metadata
        "tourney_id": ["A", "A", "A", "A", "A", "A"],
        "tourney_signature": ["a", "a", "a", "a", "a", "a"],
        "match_year": [2024, 2024, 2024, 2024, 2023, 2023],
    }
    return pd.DataFrame(data)

def test_form_all_blackbox():
    df = make_df()
    out = add_form_all(df.copy())

    # Player 1's second match should use exactly 1 past match
    p1_second = out[out["p1_id"] == 1].iloc[1]
    assert "p1_form_all_3_winrate" in p1_second
    assert p1_second["p1_form_all_3_winrate"] >= 0

def test_form_same_blackbox():
    df = make_df()
    out = add_form_same(df.copy())

    # match 3 is clay and p1 ahs no past clay matches
    p1_third = out[out["p1_id"] == 1].iloc[2]
    assert p1_third["p1_form_same_3_winrate"] == 0



def test_h2h_form_all_blackbox():
    df = make_df()
    out = add_h2h_form_all(df.copy())

    # Player 1 vs Player 2 has only 1 past H2H match
    p1_row = out[out["p1_id"] == 1].iloc[0]
    assert p1_row["p1_form_h2h_all_3_winrate"] == 0  # no past matches yet

    p1_second = out[out["p1_id"] == 1].iloc[1]
    assert p1_second["p1_form_h2h_all_3_winrate"] in (0, 1)

def test_h2h_form_same_blackbox():
    df = make_df()
    out = add_h2h_form_same(df.copy())

    # first match has no past H2H on same surface
    p1_first = out[out["p1_id"] == 1].iloc[0]
    assert p1_first["p1_form_h2h_same_3_winrate"] == 0

    # second match still no past H2H on same surface (Clay vs Hard)
    p1_second = out[out["p1_id"] == 1].iloc[1]
    assert p1_second["p1_form_h2h_same_3_winrate"] == 0




def test_tournament_form_blackbox():
    df = make_df()
    out = add_tournament_form_all(df.copy())

    # tournament-career excludes current-year matches
    p1_first = out[out["p1_id"] == 1].iloc[0]
    assert p1_first["p1_form_tournament_3_winrate"] == 0

def test_no_nans_in_form_columns():
    df = make_df()
    out = add_form_all(df)
    form_cols = [c for c in out.columns if "form_all_" in c]
    assert out[form_cols].isna().sum().sum() == 0

def test_form_columns_are_float():
    df = make_df()
    out = add_form_all(df)
    form_cols = [c for c in out.columns if "form_all_" in c]
    assert all(out[c].dtype.kind == "f" for c in form_cols)
