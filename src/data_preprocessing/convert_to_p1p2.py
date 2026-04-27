import pandas as pd

"""
convert_to_p1p2 : restructures each mirrored row into a consistent p1/p2 format,
mapping the winner to p1 and the loser to p2,
ensuring the model doesnt recieve any bias column names”
"""

def convert_to_p1p2(df):
    df = df.copy()

    # maps the winner to p1 and loser p2
    p1_cols = {
        "p1_id": "winner_id",
        "p1_rank": "winner_rank",
        "p1_age": "winner_age",
        "p1_ht": "winner_ht",
        "p1_hand": "winner_hand",
        "p1_ioc": "winner_ioc",
        "p1_rank_points": "winner_rank_points",
        "p1_ace": "w_ace",
        "p1_df": "w_df",
        "p1_svpt": "w_svpt",
        "p1_1stin": "w_1stin",
        "p1_1stwon": "w_1stwon",
        "p1_2ndwon": "w_2ndwon",
        "p1_svgms": "w_svgms",
        "p1_bpsaved": "w_bpsaved",
        "p1_bpfaced": "w_bpfaced",
        "p1_seed": "winner_seed",
        "p1_entry": "winner_entry",
        "p1_name": "winner_name",
    }

    p2_cols = {
        "p2_id": "loser_id",
        "p2_rank": "loser_rank",
        "p2_age": "loser_age",
        "p2_ht": "loser_ht",
        "p2_hand": "loser_hand",
        "p2_ioc": "loser_ioc",
        "p2_rank_points": "loser_rank_points",
        "p2_ace": "l_ace",
        "p2_df": "l_df",
        "p2_svpt": "l_svpt",
        "p2_1stin": "l_1stin",
        "p2_1stwon": "l_1stwon",
        "p2_2ndwon": "l_2ndwon",
        "p2_svgms": "l_svgms",
        "p2_bpsaved": "l_bpsaved",
        "p2_bpfaced": "l_bpfaced",
        "p2_seed": "loser_seed",
        "p2_entry": "loser_entry",
        "p2_name": "loser_name",
    }

    new_df = pd.DataFrame()

    # keep metadata
    match_cols = [ c for c in df.columns if c not in p1_cols.values() and c not in p2_cols.values() ]
    new_df[match_cols] = df[match_cols]

    # map1+p2 columns
    for new_col, old_col in p1_cols.items():
        new_df[new_col] = df[old_col] if old_col in df.columns else pd.NA

    for new_col, old_col in p2_cols.items():
        new_df[new_col] = df[old_col] if old_col in df.columns else pd.NA


    # drop leftover winner/loser and w_/l_ columns
    cols_to_drop = [
        c for c in new_df.columns
        if c.startswith("winner_")
           or c.startswith("loser_")
           or c.startswith("w_")
           or c.startswith("l_")
    ]

    new_df = new_df.drop(columns=cols_to_drop, errors="ignore")


    return new_df
