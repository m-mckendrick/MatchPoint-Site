import pandas as pd
from src.feature_engineering.elo import add_elo_features

def test_elo_blackbox_basic_columns():
    df = pd.DataFrame({
        "p1_id": [1, 1],
        "p2_id": [2, 3],
        "label": [1, 0],  # p1 beats p2, then loses to p3
        "surface_raw": ["Hard", "Clay"],
        "tourney_date": [20200101, 20200102],
        "round_encoded": [1, 2],
    })

    out = add_elo_features(df)

    assert "p1_elo_global" in out.columns
    assert "p2_elo_global" in out.columns
    assert "elo_global_diff" in out.columns

    assert "p1_elo_surface" in out.columns
    assert "p2_elo_surface" in out.columns
    assert "elo_surface_diff" in out.columns


def test_elo_blackbox_ratings_change_over_time():
    df = pd.DataFrame({
        "p1_id": [1, 1],
        "p2_id": [2, 2],
        "label": [1, 0],  # p1 beats p2, then loses to p2
        "surface_raw": ["Hard", "Hard"],
        "tourney_date": [20200101, 20200102],
        "round_encoded": [1, 2],
    })

    out = add_elo_features(df)

    #elo changes between matches
    assert out.loc[0, "p1_elo_global"] != out.loc[1, "p1_elo_global"]
    assert out.loc[0, "p2_elo_global"] != out.loc[1, "p2_elo_global"]


def test_elo_blackbox_surface_specific():
    df = pd.DataFrame({
        "p1_id": [1, 1, 1],
        "p2_id": [2, 2, 2],
        "label": [1, 1, 1],
        "surface_raw": ["Hard", "Hard", "Clay"],
        "tourney_date": [20200101, 20200102, 20200103],
        "round_encoded": [1, 2, 3],
    })

    out = add_elo_features(df)

    # surface elo should differ across surfaces
    assert out.loc[0, "p1_elo_surface"] != out.loc[1, "p1_elo_surface"]
    assert out.loc[0, "p2_elo_surface"] != out.loc[1, "p2_elo_surface"]


def test_elo_blackbox_diff_sign():
    df = pd.DataFrame({
        "p1_id": [1],
        "p2_id": [2],
        "label": [1],  # p1 wins
        "surface_raw": ["Hard"],
        "tourney_date": [20200101],
        "round_encoded": [1],
    })

    out = add_elo_features(df)

    # elo diff sign should match relative strength
    assert out.loc[0, "elo_global_diff"] == out.loc[0, "p1_elo_global"] - out.loc[0, "p2_elo_global"]
