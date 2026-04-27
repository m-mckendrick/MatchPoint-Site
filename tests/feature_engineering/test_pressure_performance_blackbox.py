import pandas as pd
import numpy as np
from src.feature_engineering.pressure_performance import add_pressure_performance_features


def test_pressure_performance_blackbox_range():
    df = pd.DataFrame([
        {
            # atp 250 3rd round
            "score": "6-2 6-2",
            "best_of": 3,
            "p1_bp_save_pct": 1/3,
            "p1_bp_conversion_pct": 1.0,
            "p2_bp_save_pct": 0.0,
            "p2_bp_conversion_pct": 0.0,
            "p1_tiebreaks_won": 0, "p2_tiebreaks_won": 0,
            "tourney_prestige": 1,        # ATP 250
            "round_points": 20,           # R32
        },
        {
            # grand slam sf
            "score": "7-6(4) 6-7(8) 6-4",
            "best_of": 3,
            "p1_bp_save_pct": 5/6,
            "p1_bp_conversion_pct": 2/5,
            "p2_bp_save_pct": 3/5,
            "p2_bp_conversion_pct": 1/6,
            "p1_tiebreaks_won": 1, "p2_tiebreaks_won": 1,
            "tourney_prestige": 4,        # Grand Slam
            "round_points": 1200,         # SF
        }
    ])

    out = add_pressure_performance_features(df.copy())


    #checks to see that the low pressure is being treated as such compared to the high pressure slam sf
    assert out.loc[0, "match_importance"] < out.loc[1, "match_importance"]
    assert out.loc[0, "stage_pressure"] < out.loc[1, "stage_pressure"]
    assert out.loc[0, "p1_pressure_index"] < out.loc[1, "p1_pressure_index"]
