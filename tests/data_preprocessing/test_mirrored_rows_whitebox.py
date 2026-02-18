import pandas as pd
from src.data_preprocessing.mirror_raw_rows import mirror_raw_rows

def test_mirror_raw_rows_whitebox(raw_match_df):

    df = raw_match_df.copy()
    mirrored = mirror_raw_rows(df)

    # first winner and loser rows
    winner_row = mirrored[mirrored["perspective"] == "winner"].iloc[0]
    loser_row = mirrored[mirrored["perspective"] == "loser"].iloc[0]

    # the pairs
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

    #check all swaps
    for w_col, l_col in player_pairs + serve_return_pairs:
        assert winner_row[w_col] == loser_row[l_col]
        assert winner_row[l_col] == loser_row[w_col]

    # columns should be identical
    metadata_cols = [
        "tourney_id", "tourney_name", "surface", "draw_size",
        "tourney_level", "tourney_date", "match_num",
        "round", "score", "minutes", "best_of"
    ]

    for col in metadata_cols:
        assert winner_row[col] == loser_row[col]

    # label checks
    assert winner_row["label"] == 1
    assert loser_row["label"] == 0
