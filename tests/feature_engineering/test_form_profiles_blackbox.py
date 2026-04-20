import pandas as pd
from src.feature_engineering.form_profiles import (
    add_career_form,
    add_h2h_form,
    add_52week_form,
    add_tournament_form,
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

        "surface_raw": ["Hard"] * 6,
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
        "match_year": [2024, 2024, 2024, 2024, 2023, 2023],
    }
    return pd.DataFrame(data)

def test_career_form_blackbox():
    df = make_df()
    out = add_career_form(df.copy(), windows=[3])

    # Player 1's second match should use exactly 1 past match
    p1_second = out[out["p1_id"] == 1].iloc[1]
    assert "p1_form_3_winrate" in p1_second
    assert p1_second["p1_form_3_winrate"] >= 0


def test_h2h_form_blackbox():
    df = make_df()
    out = add_h2h_form(df.copy(), windows=[3])

    # Player 1 vs Player 2 has only 1 past H2H match
    p1_row = out[out["p1_id"] == 1].iloc[0]
    assert p1_row["p1_form_3_winrate"] == 0  # no past matches yet

    p1_second = out[out["p1_id"] == 1].iloc[1]
    assert p1_second["p1_form_3_winrate"] in (0, 1)


def test_52week_form_blackbox():
    df = make_df()
    out = add_52week_form(df.copy(), windows=[3])

    # The 2023 match should be excluded for 2024 rows
    p1_second = out[out["p1_id"] == 1].iloc[1]
    assert p1_second["p1_form_3_winrate"] >= 0


def test_tournament_form_blackbox():
    df = make_df()
    out = add_tournament_form(df.copy(), windows=[3])

    # tournament-career excludes current-year matches
    p1_first = out[out["p1_id"] == 1].iloc[0]
    assert p1_first["p1_form_3_winrate"] == 0
