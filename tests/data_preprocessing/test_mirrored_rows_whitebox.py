import pandas as pd
from src.data_preprocessing.mirror_rows import mirror_rows

def test_mirror_rows_whitebox(cleaned_match_df):

    df = cleaned_match_df.copy()
    mirrored = mirror_rows(df)

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
        ("w_1stin", "l_1stin"),
        ("w_1stwon", "l_1stwon"),
        ("w_2ndwon", "l_2ndwon"),
        ("w_svgms", "l_svgms"),
        ("w_bpsaved", "l_bpsaved"),
        ("w_bpfaced", "l_bpfaced"),
    ]

    #check all swaps
    #check if both are na if so do nothing
    for w_col, l_col in player_pairs + serve_return_pairs:
        if pd.isna(winner_row[w_col]) and pd.isna(loser_row[l_col]):
            pass
        else:
            assert winner_row[w_col] == loser_row[l_col]

        # reverse swap (loser → winner)
        if pd.isna(winner_row[l_col]) and pd.isna(loser_row[w_col]):
            pass
        else:
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
