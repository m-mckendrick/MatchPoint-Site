import pandas as pd
from src.data_preprocessing.convert_to_p1p2 import convert_to_p1p2

def test_all_p1_columns_present():
    df = pd.DataFrame([{
        "winner_id": 1,
        "winner_rank": 10,
        "winner_age": 25,
        "winner_ht": 190,
        "winner_hand": "Right",
        "winner_ioc": "USA",
        "winner_rank_points": 3000,
        "w_ace": 12,
        "w_df": 3,
        "w_svpt": 80,
        "w_1stin": 50,
        "w_1stwon": 35,
        "w_2ndwon": 20,
        "w_svgms": 12,
        "w_bpsaved": 4,
        "w_bpfaced": 6,

        "loser_id": 2,
        "loser_rank": 30,
        "loser_age": 22,
        "loser_ht": 185,
        "loser_hand": "Left",
        "loser_ioc": "ESP",
        "loser_rank_points": 1500,
        "l_ace": 4,
        "l_df": 2,
        "l_svpt": 70,
        "l_1stin": 40,
        "l_1stwon": 25,
        "l_2ndwon": 15,
        "l_Svgms": 10,
        "l_bpsaved": 3,
        "l_bpfaced": 5,
    }])

    out = convert_to_p1p2(df)

    assert out.loc[0, "p1_id"] == 1
    assert out.loc[0, "p1_rank"] == 10
    assert out.loc[0, "p1_age"] == 25
    assert out.loc[0, "p1_ace"] == 12
    assert out.loc[0, "p1_df"] == 3

def test_no_winner_loser_columns_remain():
    df = pd.DataFrame([{
        "winner_id": 1,
        "loser_id": 2,
        "winner_rank": 10,
        "loser_rank": 20,
        "w_ace": 5,
        "l_ace": 2
    }])

    out = convert_to_p1p2(df)

    forbidden = [c for c in out.columns if "winner" in c or "loser" in c or c.startswith("w_") or c.startswith("l_")]
    assert forbidden == []
