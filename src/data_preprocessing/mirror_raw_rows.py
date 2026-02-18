import pandas as pd

#for logistical regression need a winner perspective 1 and losers perspective 0
def mirror_raw_rows(df: pd.DataFrame) -> pd.DataFrame:

    df_loser = df.copy()

    #player attribute pairs
    player_pairs = [
        ("winner_id", "loser_id"),
        ("winner_seed", "loser_seed"),
        ("winner_entry", "loser_entry"),
        ("winner_name", "loser_name"),
        ("winner_hand", "loser_hand"),
        ("winner_ht", "loser_ht"),
        ("winner_ioc", "loser_ioc"),
        ("winner_age", "loser_age"),
        ("winner_rank", "loser_rank"),
        ("winner_rank_points", "loser_rank_points"),
    ]

    #serve/return pairs
    serve_return_pairs = [
        ("w_ace", "l_ace"),
        ("w_df", "l_df"),
        ("w_svpt", "l_svpt"),
        ("w_1stIn", "l_1stIn"),
        ("w_1stWon", "l_1stWon"),
        ("w_2ndWon", "l_2ndWon"),
        ("w_SvGms", "l_SvGms"),
        ("w_bpSaved", "l_bpSaved"),
        ("w_bpFaced", "l_bpFaced"),
    ]

    #swap all pairs
    all_pairs = player_pairs + serve_return_pairs

    for w_col, l_col in all_pairs:
        df_loser[w_col] = df[l_col] 
        df_loser[l_col] = df[w_col]


    # adding perspective and label for model
    df["perspective"] = "winner"
    df["label"] = 1

    df_loser["perspective"] = "loser"
    df_loser["label"] = 0


    # concatonate the winner and loser perspective pds
    df_combined = pd.concat([df, df_loser], ignore_index=True)

    return df_combined
