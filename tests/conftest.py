import pandas as pd
import pytest
from datetime import datetime


#raw df example for testing
#inclueds 2 matches before cleaning
@pytest.fixture
def raw_match_df():
    return pd.DataFrame([
        {
            "tourney_id": "2024-580",
            "tourney_name": "Australian Open",
            "surface": "Hard",
            "draw_size": 128,
            "tourney_level": "G",
            "tourney_date": 20240115,
            "match_num": 131,

            "winner_id": 126094,
            "winner_seed": 5,
            "winner_entry": "",
            "winner_name": "Andrey Rublev",
            "winner_hand": "R",
            "winner_ht": 188,
            "winner_ioc": "RUS",
            "winner_age": 26.2,

            "loser_id": 205734,
            "loser_seed": "",
            "loser_entry": "",
            "loser_name": "Thiago Seyboth Wild",
            "loser_hand": "R",
            "loser_ht": 185,
            "loser_ioc": "BRA",
            "loser_age": 23.8,

            "score": "7-5 6-4 3-6 4-6 7-6(6)",
            "best_of": 5,
            "round": "R128",
            "minutes": 222,

            # winner serve-return stats
            "w_ace": 21,
            "w_df": 1,
            "w_svpt": 154,
            "w_1stIn": 100,
            "w_1stWon": 83,
            "w_2ndWon": 25,
            "w_SvGms": 27,
            "w_bpSaved": 4,
            "w_bpFaced": 8,

            #loser serve-return stats
            "l_ace": 15,
            "l_df": 6,
            "l_svpt": 202,
            "l_1stIn": 130,
            "l_1stWon": 91,
            "l_2ndWon": 34,
            "l_SvGms": 26,
            "l_bpSaved": 19,
            "l_bpFaced": 23,

            "winner_rank": 5,
            "winner_rank_points": 5010,
            "loser_rank": 77,
            "loser_rank_points": 718,
        },
        {
            "tourney_id": "2024-580",
            "tourney_name": "Australian Open",
            "surface": "Hard",
            "draw_size": 128,
            "tourney_level": "G",
            "tourney_date": 20240115,
            "match_num": 211,

            "winner_id": 207989,
            "winner_seed": 2,
            "winner_entry": "",
            "winner_name": "Carlos Alcaraz",
            "winner_hand": "R",
            "winner_ht": 183,
            "winner_ioc": "ESP",
            "winner_age": 20.6,

            "loser_id": 209992,
            "loser_seed": "",
            "loser_entry": "WC",
            "loser_name": "Juncheng Shang",
            "loser_hand": "L",
            "loser_ht": 180,
            "loser_ioc": "CHN",
            "loser_age": 18.9,

            "score": "6-1 6-1 1-0 RET",
            "best_of": 5,
            "round": "R32",
            "minutes": 66,

            "w_ace": 3,
            "w_df": 1,
            "w_svpt": 34,
            "w_1stIn": 19,
            "w_1stWon": 16,
            "w_2ndWon": 13,
            "w_SvGms": 8,
            "w_bpSaved": 0,
            "w_bpFaced": 0,

            "l_ace": 1,
            "l_df": 0,
            "l_svpt": 50,
            "l_1stIn": 26,
            "l_1stWon": 13,
            "l_2ndWon": 7,
            "l_SvGms": 7,
            "l_bpSaved": 6,
            "l_bpFaced": 12,

            "winner_rank": 2,
            "winner_rank_points": 8855,
            "loser_rank": 140,
            "loser_rank_points": 435,
        }
    ])


#cleaned df with matches that have been cleaned
@pytest.fixture
def cleaned_match_df(raw_match_df):
    df = raw_match_df.copy()

    #header normalisation
    df.columns = df.columns.str.lower()

    #convert date
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d")

    #map hands
    hand_map = {"R": "Right", "L": "Left", "U": "Unknown"}
    df["winner_hand"] = df["winner_hand"].map(hand_map)
    df["loser_hand"] = df["loser_hand"].map(hand_map)

    #normalise categories
    df["surface"] = df["surface"].str.title()
    df["round"] = df["round"].str.upper()

    #add tiebreak scores (placeholder)
    df["tiebreak_scores"] = ["none" "none" "none" "none" "none","none none none"]

    return df
