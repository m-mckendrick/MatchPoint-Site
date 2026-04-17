import pandas as pd
import numpy as np
from src.feature_engineering.pressure_performance import add_pressure_performance_features


def test_pressure_performance_blackbox_range():
    df = pd.DataFrame([
        {
            # atp 250 3rd round
            "score": "6-2 6-2",
            "best_of": 3,
            "p1_bpsaved": 1, "p1_bpfaced": 3,
            "p2_bpsaved": 0, "p2_bpfaced": 1,
            "p1_tiebreaks_won": 0, "p2_tiebreaks_won": 0,
            "tourney_prestige": 1,        # ATP 250
            "round_points": 20,           # R32
        },
        {
            # grand slam sf
            "score": "7-6(4) 6-7(8) 6-4",
            "best_of": 3,
            "p1_bpsaved": 5, "p1_bpfaced": 6,
            "p2_bpsaved": 3, "p2_bpfaced": 5,
            "p1_tiebreaks_won": 1, "p2_tiebreaks_won": 1,
            "tourney_prestige": 4,        # Grand Slam
            "round_points": 1200,         # SF
        }
    ])

    out = add_pressure_performance_features(df.copy())


    #checks to see that the low pressure is being treated as such compared to the high pressure slam sf
    assert out.loc[0, "match_importance"] < out.loc[1, "match_importance"]
    assert out.loc[0, "pressure_multiplier"] < out.loc[1, "pressure_multiplier"]
    assert out.loc[0, "p1_pressure_index"] < out.loc[1, "p1_pressure_index"]
    assert np.isclose(out["stage_pressure"], out["pressure_multiplier"]).all()
