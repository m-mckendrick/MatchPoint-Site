import pandas as pd
from src.feature_engineering.elo import (
    add_elo_features,
    _expected_score,
    _update_elo,
    BASE_ELO,
    K_FACTOR,
    SURFACE_K_FACTOR,
)

def test_elo_whitebox_expected_score():
    # If ratings equal → expected = 0.5
    assert abs(_expected_score(1500, 1500) - 0.5) < 1e-6

    # Higher rating → higher expected score
    assert _expected_score(1600, 1400) > 0.5
    assert _expected_score(1400, 1600) < 0.5


def test_elo_whitebox_update_logic():
    r = BASE_ELO
    expected = 0.5
    new_r = _update_elo(r, score=1.0, expected=expected, k=K_FACTOR)

    # winning should increase rating
    assert new_r > r

    new_r2 = _update_elo(r, score=0.0, expected=expected, k=K_FACTOR)

    # losing should decrease rating
    assert new_r2 < r


def test_elo_whitebox_chronological_ordering():
    df = pd.DataFrame({
        "p1_id": [1, 1],
        "p2_id": [2, 2],
        "label": [1, 1],
        "surface_raw": ["Hard", "Hard"],
        "tourney_date": [20200102, 20200101],  # out of order
        "round_encoded": [2, 1],
    })

    out = add_elo_features(df)

    # first processed match should have base Elo
    assert out.loc[0, "p1_elo_global"] == BASE_ELO
    assert out.loc[0, "p2_elo_global"] == BASE_ELO

    # second match should have updated Elo
    assert out.loc[1, "p1_elo_global"] != BASE_ELO


def test_elo_whitebox_surface_isolation():
    df = pd.DataFrame({
        "p1_id": [1, 1, 1],
        "p2_id": [2, 2, 2],
        "label": [1, 1, 1],
        "surface_raw": ["Hard", "Hard", "Clay"],
        "tourney_date": [20200101, 20200102, 20200103],
        "round_encoded": [1, 2, 3],
    })

    out = add_elo_features(df)

    # hard elo should not equal clay elo
    assert out.loc[0, "p1_elo_surface"] != out.loc[1, "p1_elo_surface"]
    assert out.loc[0, "p2_elo_surface"] != out.loc[1, "p2_elo_surface"]
