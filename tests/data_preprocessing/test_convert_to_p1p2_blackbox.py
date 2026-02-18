import pandas as pd
from src.data_preprocessing.convert_to_p1p2 import convert_to_p1p2

def test_convert_basic_mapping(raw_match_df):
    out = convert_to_p1p2(raw_match_df)

    # loop through fixture
    for i in range(len(raw_match_df)):
        row_in = raw_match_df.iloc[i]
        row_out = out.iloc[i]

        # p1 should match winner
        assert row_out["p1_id"] == row_in["winner_id"]
        assert row_out["p1_rank"] == row_in["winner_rank"]
        assert row_out["p1_ace"] == row_in["w_ace"]

        # p2 should match loser
        assert row_out["p2_id"] == row_in["loser_id"]
        assert row_out["p2_rank"] == row_in["loser_rank"]
        assert row_out["p2_ace"] == row_in["l_ace"]


def test_metadata_preserved(raw_match_df):
    out = convert_to_p1p2(raw_match_df)

    metadata_cols = [
        "tourney_id", "tourney_name", "surface", "draw_size",
        "tourney_level", "tourney_date", "match_num",
        "score", "best_of", "round", "minutes"
    ]

    for col in metadata_cols:
        assert all(out[col] == raw_match_df[col])


def test_output_shape(raw_match_df):
    out = convert_to_p1p2(raw_match_df)

    # same number of rows
    assert len(out) == len(raw_match_df)

    # p1/p2 columns exist
    expected_p1_cols = ["p1_id", "p1_rank", "p1_ace"]
    expected_p2_cols = ["p2_id", "p2_rank", "p2_ace"]

    for col in expected_p1_cols + expected_p2_cols:
        assert col in out.columns


def test_no_winner_loser_columns(raw_match_df):
    out = convert_to_p1p2(raw_match_df)
    #make sure no left over w l columns
    forbidden = [
        c for c in out.columns
        if c.startswith("winner_")
           or c.startswith("loser_")
           or c.startswith("w_")
           or c.startswith("l_")
    ]

    assert forbidden == []
