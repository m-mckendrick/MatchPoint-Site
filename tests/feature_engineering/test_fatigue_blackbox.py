import pandas as pd
import numpy as np

from src.feature_engineering.fatigue import add_fatigue_index

def test_fatigue_blackbox_end_to_end():
    df = pd.DataFrame({
        "p1_id": [1, 1, 1],
        "p2_id": [2, 2, 2],
        "tourney_date": pd.to_datetime([
            "2024-01-01",
            "2024-01-02",
            "2024-01-10"
        ]),
        "match_num": [1, 2, 3],
        "minutes": [120, 90, 60]
    })

    out = add_fatigue_index(df, tau=3, min_effect=0.5)

    # no history
    assert out.loc[0, "p1_fatigue_raw"] == 0

    # fatigue from match 1 only
    assert out.loc[1, "p1_fatigue_raw"] > 0

    # match 1 should be ignored , match 2 should count
    assert out.loc[2, "p1_fatigue_raw"] > 0
    assert out.loc[2, "p1_fatigue_raw"] < out.loc[1, "p1_fatigue_raw"]
