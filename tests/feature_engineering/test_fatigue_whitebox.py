import pandas as pd
import numpy as np

from src.feature_engineering.fatigue import add_fatigue_features

def test_fatigue_whitebox_basic_decay():
    df = pd.DataFrame({
        "p1_id": [1, 1],
        "p2_id": [2, 2],
        "tourney_date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
        "round_encoded": [0, 1,],
        "minutes": [100, 50],
        "tourney_signature": ["Adelaide_A_Hard"] * 2
    })

    out = add_fatigue_features(df, tau=3, min_effect=0.0001)

    # first match has no history so fatigue = 0
    assert out.loc[0, "p1_fatigue_in"] == 0

    # second match should have fatigue from match 1
    round_gap = 1 - 0
    raw = 100 * np.exp(-round_gap/ 3)
    expected = raw/raw
    assert abs(out.loc[1, "p1_fatigue_in"] - expected) < 1e-6

    # same tournament so no pre‑tournament fatigue
    assert out.loc[0, "p1_fatigue_pre"] == 0
    assert out.loc[1, "p1_fatigue_pre"] == 0