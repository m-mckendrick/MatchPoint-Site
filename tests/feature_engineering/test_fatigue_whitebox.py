import pandas as pd
import numpy as np

from src.feature_engineering.fatigue import add_fatigue_index

def test_fatigue_whitebox_basic_decay():
    df = pd.DataFrame({
        "p1_id": [1, 1],
        "p2_id": [2, 2],
        "tourney_date": pd.to_datetime(["2024-01-01", "2024-01-03"]),
        "match_num": [1, 2],
        "minutes": [100, 50]
    })

    out = add_fatigue_index(df, tau=3, min_effect=0.0001)

    # first match has no history so fatigue = 0
    assert out.loc[0, "p1_fatigue_index"] == 0

    # second match should have fatigue from match 1
    raw = 100 * np.exp(-2 / 3)
    expected = raw/raw
    assert abs(out.loc[1, "p1_fatigue_index"] - expected) < 1e-6
