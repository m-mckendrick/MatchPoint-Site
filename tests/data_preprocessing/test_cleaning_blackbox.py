import pandas as pd

from datetime import datetime

from src.data_preprocessing.clean_pipeline import (
    normalise_headers,
    parse_dates,
    convert_numeric,
    clean_ranks_and_seeds,
    map_hands,
    normalize_tourney_names,
    clean_categories,
    filter_atp_events,
    filter_matches,
    filter_rr_events,
    clean_matches,
    remove_retirement_and_flag,
    extract_tiebreak_score_string_row,
    drop_scoreless_retirements
)


#black-box tests
#test output of functions

#testing header to lowercase etc
def test_normalise_headers_basic():
    df = pd.DataFrame(columns=[" winner Rank ", "loser-rank"])
    out = normalise_headers(df)
    assert list(out.columns) == ["winner_rank", "loser_rank"]

#testing date are turned to datatime correctly
def test_parse_dates_valid_and_invalid():
    df = pd.DataFrame({"tourney_date": ["20240101", "bad"]})
    out = parse_dates(df)
    assert pd.isna(out["tourney_date"].iloc[1])
    assert out["tourney_date"].iloc[0].year == 2024


#testing teh numeric convert for all the stats
def test_convert_numeric():
    df = pd.DataFrame({
        "w_ace": ["10", "bad"],
        "l_df": ["3", None]
    })
    out = convert_numeric(df)
    assert out["w_ace"].dtype == "float64"
    assert pd.isna(out["w_ace"].iloc[1])


#all rankings and seed to integers
def test_clean_ranks_and_seeds():
    df = pd.DataFrame({
        "winner_seed": ["1", None],
        "loser_seed": ["2", "3"],
        "winner_rank": ["10", "20"],
        "loser_rank": ["30", None],
        "winner_rank_points": ["1000", "2000"],
        "loser_rank_points": ["500", None]
    })
    out = clean_ranks_and_seeds(df)
    assert str(out["winner_seed"].dtype) == "Int64"
    assert pd.isna(out["winner_seed"].iloc[1])

#test r == right, u == unknown
def test_map_hands():
    df = pd.DataFrame({"winner_hand": ["R"], "loser_hand": ["U"]})
    out = map_hands(df)
    assert out["winner_hand"].iloc[0] == "Right"
    assert out["loser_hand"].iloc[0] == "Unknown"


def test_clean_categories():
    df = pd.DataFrame({"surface": ["hard"], "round": ["qf"]})
    out = clean_categories(df)
    assert out["surface"].iloc[0] == "Hard"
    assert out["round"].iloc[0] == "QF"

#testing laver cup is removed and aussie open kept
def test_filter_atp_events():
    df = pd.DataFrame({
        "tourney_level": ["A", "X", "F" ],
        "tourney_name": ["Australian Open", "Laver Cup", "NextGen Finals"]
    })
    out = filter_atp_events(df)
    assert len(out) == 1
    assert out["tourney_level"].iloc[0] == "A"

def test_filter_matches_only_removes_br_er():
    df = pd.DataFrame({
        "round": ["R32", "R16", "QF", "SF", "F", "RR", "BR", "ER"]
    })

    out = filter_matches(df)

    # BR and ER must be removed
    assert "BR" not in out["round"].values
    assert "ER" not in out["round"].values

    expected = {"R32", "R16", "QF", "SF", "F", "RR"}
    assert set(out["round"].values) == expected

def test_filter_rr_events_removes_non_finals_rr():

    df = pd.DataFrame([
        # RR that leave
        {"tourney_name": "Dusseldorf", "round": "RR"},
        {"tourney_name": "Las Vegas", "round": "RR"},
        {"tourney_name": "Buenos Aires", "round": "RR"},

        # RR that stay
        {"tourney_name": "Tour Finals", "round": "RR"},
        {"tourney_name": "Masters Cup", "round": "RR"},
        {"tourney_name": "ATP Finals", "round": "RR"},

        # normal rounds
        {"tourney_name": "Wimbledon", "round": "R32"},
    ])

    cleaned = filter_rr_events(df)

    expected_tourneys = {
        "Tour Finals",
        "Masters Cup",
        "ATP Finals",
        "Wimbledon",
    }
    assert set(cleaned["tourney_name"].unique()) == expected_tourneys

    # RR only appear in ATP Finals
    rr_rows = cleaned[cleaned["round"] == "RR"]
    assert set(rr_rows["tourney_name"].unique()) == {
        "Tour Finals",
        "Masters Cup",
        "ATP Finals"
    }

    # round rows should remain untouched
    assert "Wimbledon" in cleaned["tourney_name"].values
    assert cleaned.loc[cleaned["tourney_name"] == "Wimbledon", "round"].iloc[0] == "R32"




def test_remove_retirement_and_flag_basic():
    df = pd.DataFrame({
        "score": [
            "7-5 4-0 RET",
            "6-4 6-4",
            "6-3 RET.",
            "7-6(5) 2-1 RETIREMENT",
            "W/O",
            None
        ]
    })

    out = remove_retirement_and_flag(df.copy())

    assert out["is_retirement"].tolist() == [1, 0, 1, 1, 1, 0]

    assert out["score"].tolist() == [
        "7-5 4-0",
        "6-4 6-4",
        "6-3",
        "7-6(5) 2-1",
        "",
        "0-0"
    ]

#testing all iterations of tiebreaks scores are calculated correctly before added to csv
def test_classic_tiebreak_simple():
    row = {
        "score": "7-6(3) 6-4",
        "best_of": 3,
        "tourney_name": "Queens",
        "tourney_date": datetime(2018, 7, 1),
        "tourney_level": "A"
    }
    out = extract_tiebreak_score_string_row(row)
    assert out == "7-3 none"


def test_classic_tiebreak_extended():
    row = {
        "score": "7-6(8) 6-4",
        "best_of": 3,
        "tourney_name": "Queens",
        "tourney_date": datetime(2018, 7, 1),
        "tourney_level": "A"
    }
    out = extract_tiebreak_score_string_row(row)
    assert out == "10-8 none"


def test_no_tiebreaks():
    row = {
        "score": "6-4 6-4",
        "best_of": 3,
        "tourney_name": "Queens",
        "tourney_date": datetime(2018, 7, 1),
        "tourney_level": "A"
    }
    out = extract_tiebreak_score_string_row(row)
    assert out == "none none"


def test_advantage_set():
    row = {
        "score": "13-11",
        "best_of": 5,
        "tourney_name": "Wimbledon",
        "tourney_date": datetime(2018, 7, 1),
        "tourney_level": "G"
    }
    out = extract_tiebreak_score_string_row(row)
    assert out == "none"


def test_match_tiebreak_post_2022():
    row = {
        "score": "6-4 6-4 4-6 4-6 7-6(8)",
        "best_of": 5,
        "tourney_name": "Australian Open",
        "tourney_date": datetime(2023, 1, 20),
        "tourney_level": "G"
    }
    out = extract_tiebreak_score_string_row(row)
    assert out == "none none none none 10-8"


def test_ao_2019_to_2021_match_tiebreak():
    row = {
        "score": "6-4 6-4 4-6 4-6 7-6(12)",
        "best_of": 5,
        "tourney_name": "Australian Open",
        "tourney_date": datetime(2020, 1, 20),
        "tourney_level": "G"
    }
    out = extract_tiebreak_score_string_row(row)
    assert out == "none none none none 14-12"


#testing full pipeline/(not testing convert to model friendly)
def test_clean_matches_full_pipeline(raw_match_df):
    out = clean_matches(raw_match_df)

    # Check that key transformations happened
    assert out["surface"].iloc[0] == "Hard"
    assert out["winner_hand"].iloc[0] == "Right"
    assert out["winner_rank"].dtype == "Int64"
    assert out["tourney_date"].iloc[0].year == 2024

#found out real function isn't robust to handle None (None isn't actually passed in teh file because there not None score)
def test_drop_scoreless_retirements():
    df = pd.DataFrame({
        "score": [" RET", " ret", "6-4 2-1 RET", "7-6(5)"]
    })

    out = drop_scoreless_retirements(df)

    # should remove RET and no digits with ret cases
    assert " RET" not in out["score"].values
    assert " ret" not in out["score"].values

    # should keep normal matches and normal retirements
    assert "6-4 2-1 RET" in out["score"].values
    assert "7-6(5)" in out["score"].values

#this test not likely to be super useful
def test_remove_retirement_and_flag_nan_score():
    df = pd.DataFrame({"score": [None]})
    out = remove_retirement_and_flag(df)
    assert out["is_retirement"].iloc[0] == 0
    assert out["score"].iloc[0] == "0-0"

def test_remove_retirement_and_flag_keeps_normal_scores():
    df = pd.DataFrame({"score": ["6-4 6-4"]})
    out = remove_retirement_and_flag(df)
    assert out["is_retirement"].iloc[0] == 0
    assert out["score"].iloc[0] == "6-4 6-4"


def test_normalize_tourney_names_blackbox():
    # input dataframe with known inconsistencies
    df = pd.DataFrame({
        "tourney_name": [
            "Belgrade ",
            "Belgrade",
            "Rio De Janeiro",
            "Rio de Janeiro",
            "St. Petersburg",
            "St Petersburg",
            "Us Open",
            "US Open",
            "Wimbledon",  # control case
        ]
    })

    expected = [
        "Belgrade",
        "Belgrade",
        "Rio de Janeiro",
        "Rio de Janeiro",
        "St Petersburg",
        "St Petersburg",
        "US Open",
        "US Open",
        "Wimbledon",
    ]

    cleaned = normalize_tourney_names(df.copy())

    assert cleaned["tourney_name"].tolist() == expected

    assert all(not name.endswith(" ") for name in cleaned["tourney_name"])

    assert cleaned.loc[8, "tourney_name"] == "Wimbledon"
