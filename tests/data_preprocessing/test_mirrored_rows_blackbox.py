import pandas as pd
from src.data_preprocessing.mirror_rows import mirror_rows

def test_mirror_rows_blackbox(cleaned_match_df):

    df = cleaned_match_df.copy()
    mirrored = mirror_rows(df)

    # row count doubled
    assert len(mirrored) == len(df) * 2

    # perspectives present
    assert set(mirrored["perspective"]) == {"winner", "loser"}

    # labels correct
    assert set(mirrored["label"]) == {0, 1}

    # id swap on first row
    original = mirrored[mirrored["perspective"] == "winner"].iloc[0]
    mirrored_row = mirrored[mirrored["perspective"] == "loser"].iloc[0]

    assert original["winner_id"] == mirrored_row["loser_id"]
    assert original["loser_id"] == mirrored_row["winner_id"]
