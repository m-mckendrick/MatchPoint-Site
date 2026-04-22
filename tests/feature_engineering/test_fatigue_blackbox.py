import pandas as pd
import numpy as np

from src.feature_engineering.fatigue import add_fatigue_features

def test_fatigue_blackbox_end_to_end():
    df = pd.DataFrame({
        "p1_id": [1, 1, 1, 1],
        "p2_id": [2, 2, 2, 2],
        "tourney_date": pd.to_datetime([
            "2024-01-01",
            "2024-01-01",
            "2024-01-15",
            "2024-01-15"
        ]),

        "tourney_signature": ["A", "A", "B", "B"],
        "round_encoded": [0, 1, 0, 1],
        "minutes": [120, 90, 60, 30]
    })

    out = add_fatigue_features(df, tau_pre=3, tau_in=3, min_effect=0.01)

    # columns exist
    assert "p1_fatigue_pre" in out.columns
    assert "p1_fatigue_in" in out.columns

    # no nones
    assert out["p1_fatigue_pre"].isna().sum() == 0
    assert out["p1_fatigue_in"].isna().sum() == 0

    #in tournament
    assert out.loc[0, "p1_fatigue_in"] == 0
    assert out.loc[1, "p1_fatigue_in"] > 0
    assert out.loc[2, "p1_fatigue_in"] == 0
    assert out.loc[3, "p1_fatigue_in"] > 0

    #pre tournament
    assert out.loc[0, "p1_fatigue_pre"] == 0
    assert out.loc[1, "p1_fatigue_pre"] == 0
    assert out.loc[2, "p1_fatigue_pre"] > 0
    assert out.loc[3, "p1_fatigue_pre"] > 0

    # pre‑tournament fatigue constant within tournament
    assert out.loc[2, "p1_fatigue_pre"] == out.loc[3, "p1_fatigue_pre"]

    # normalisation
    assert out["p1_fatigue_pre"].max() <= 1
    assert out["p1_fatigue_in"].max() <= 1