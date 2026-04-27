# Data Cleaning and Validation

# settings
import pandas as pd
import numpy as np
from pathlib import Path

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)

from data.loader import RAW_MATCHES_PATH, MATCHES_PATH


"""
step 1 : Normalise Headers, Values and Missing Values
"""
def normalise_headers(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace("-", "_")
    return df

def normalise_missing_values(df):
    #with NA
    missing_markers = ["", " ", ".", "--", "NR", "?", "n/a", "N/A"]
    df = df.replace(missing_markers, pd.NA)
    return df

def clean_categories(df):
    df["surface"] = df["surface"].str.title()
    df["round"] = df["round"].str.upper()
    return df

def normalize_tourney_names(df):
    # strip whitespace
    df["tourney_name"] = df["tourney_name"].str.strip()

    # fix specific known inconsistencies
    df["tourney_name"] = df["tourney_name"].replace({
        "Belgrade ": "Belgrade",
        "belgrade ": "Belgrade",   # safety
        "Rio De Janeiro": "Rio de Janeiro",
        "St. Petersburg": "St Petersburg",
        "Us Open": "US Open",
    })

    return df


"""
step 2 : Type Conversion
turn pandas types to usable types
"""
# tourney date int -> datetime
def parse_dates(df):
    df["tourney_date"] = pd.to_datetime( df["tourney_date"], format="%Y%m%d", errors="coerce" )
    return df

# serve stats -> numeric
def convert_numeric(df):
    serve_cols = [c for c in df.columns if c.startswith(("w_", "l_"))]
    df[serve_cols] = df[serve_cols].apply(pd.to_numeric, errors="coerce")
    return df

# all rankings and seedings -> integers
def clean_ranks_and_seeds(df):
    seed_cols = ["winner_seed", "loser_seed"]
    rank_cols = ["winner_rank", "loser_rank", "winner_rank_points", "loser_rank_points"]

    df[seed_cols + rank_cols] = df[seed_cols + rank_cols].replace("", pd.NA)

    df[seed_cols] = df[seed_cols].astype("Int64")
    df[rank_cols] = df[rank_cols].astype("Int64")
    return df


"""
step 3 : Value Mapping (help model)
"""

def map_hands(df):
    hand_map = {"R": "Right", "L": "Left", "U": "Unknown"}
    df["winner_hand"] = df["winner_hand"].map(hand_map)
    df["loser_hand"] = df["loser_hand"].map(hand_map)
    return df


"""
step 4 : remove non-matches, and incorrect rows
includes removing match that were retirement before 1 game was completed 
"""

def remove_non_matches(df):
    bad_keywords = ["W/O", "DEF", "Def.", "ABN", "SUSP", "WO"]
    pattern = "|".join(bad_keywords)
    return df[~df["score"].str.contains(pattern, na=False)]

def drop_scoreless_retirements(df):
    retirement_tokens = {"RET", "RET.", "RETIREMENT"}
    score_str = df["score"].astype(str).str.upper()

    def is_scoreless_retirement(s):
        # contains a retirement token
        has_ret = any(tok in s for tok in retirement_tokens)
        # contains no digits
        has_no_score = not any(ch.isdigit() for ch in s)
        return has_ret and has_no_score

    mask = score_str.apply(is_scoreless_retirement)

    #remove rows that has RET and no Digits

    return df[~mask].copy()

# drop if doesn't have synthetic row essential columns
def drop_incomplete_rows(df):
    essential = ["winner_name", "loser_name", "tourney_date", "surface", "round"]
    return df.dropna(subset=essential)

def remove_duplicates(df):
    return df.drop_duplicates(subset=["tourney_id", "match_num"])



#%%
"""
step 5 : data validation
"""

def validate_score_format(df):
    bad = df[~df["score"].str.match(r"^[0-9\-() ]+$", na=False)]
    if len(bad) > 0:
        print("Warning: unexpected score formats:", bad["score"].unique())
    return df

# no dupes p1 vs p1
def validate_winner_loser(df):
    bad = df[df["winner_name"] == df["loser_name"]]
    if len(bad) > 0:
        print("Warning: identical winner/loser names found.")
    return df

# just in case of crazy value
def validate_serve_stats(df):
    bad = df[df["w_ace"] > df["w_svpt"]]
    if len(bad) > 0:
        print("Warning: impossible serve stats detected (aces > serve points).")
    return df

def validate_physical_stats(df):
    df = df[df["winner_ht"].between(140, 220) | df["winner_ht"].isna()]
    df = df[df["loser_ht"].between(140, 220) | df["loser_ht"].isna()]
    df = df[df["winner_age"].between(14, 45) | df["winner_age"].isna()]
    df = df[df["loser_age"].between(14, 45) | df["loser_age"].isna()]
    return df


def validate_surface(df):
    valid = {"Hard", "Clay", "Grass", "Carpet"}
    bad = df[~df["surface"].isin(valid)] #bad = all surfaces not in above list
    if len(bad) > 0:
        print("Warning: unexpected surface values found:")
        print(bad["surface"].unique())
    return df

def validate_round(df):
    valid = {"R128","R64","R32","R16","QF","SF","F","RR"}
    bad = df[~df["round"].isin(valid)]
    if len(bad) > 0:
        print("Warning: unexpected round values found:")
        print(bad["round"].unique())
    return df

def validate_best_of(df):
    bad = df[~df["best_of"].isin([3,5])]
    if len(bad) > 0:
        print("Warning: unexpected best_of values found:")
        print(bad["best_of"].unique())
    return df

def validate_minutes(df):
    bad = df[(df["minutes"] < 20) & df["minutes"].notna()] # length not under 20 unless null
    if len(bad) > 0:
        print("Warning: unusually short matches detected.")
    bad = df[(df["minutes"] > 400)]
    if len(bad) > 0:
        print("Warning: unusually long matches detected.")
    return df



"""
step 6 : remove all matches that are not atp events
also remove matches that are early round (qualifying) and bronze medal matches
"""

def filter_atp_events(df):
    allowed_levels = {"A", "G", "M", "F"}
    df = df[df["tourney_level"].isin(allowed_levels)]

    banned_keywords = [
        "United Cup",
        "Laver Cup",
        "Next Gen Finals",
        'NextGen Finals',
        "Olympics",
        "Davis Cup",
        "Exhibition"
    ]

    pattern = "|".join(banned_keywords)
    df = df[~df["tourney_name"].str.contains(pattern, case=False, na=False)]

    return df

def filter_matches(df):
    banned_rounds = {"BR", "ER"}
    return df[~df["round"].isin(banned_rounds)]

def filter_rr_events(df):
    # keep RR only for ATP finals as otherwise they are worth 0 points
    rr_bad = df[(df["round"] == "RR") &
                (~df["tourney_name"].isin(["Masters Cup", "Tour Finals", "ATP Finals"]))]
    return df.drop(rr_bad.index)


#%%
"""
step 7 : add tiebreak scores to data
includes test for type of tiebreak logic 
also removes RET from score string to make score parsing simpler
"""

def remove_retirement_and_flag(df):
    retirement_tokens = {"RET", "RET.", "RETIREMENT", "W/O", "WO"}

    retirement_flags = []
    cleaned_scores = []

    #if there's no score
    #after testing need to make sure score is a string
    for score in df["score"]:
        #if score not string (Nan ect)
        if not isinstance(score, str):
            # retirement with no score
            retirement_flags.append(0)
            cleaned_scores.append("0-0")
            continue

        sets = score.split()
        cleaned = []
        retired = 0

        for set in sets:
            if set.upper() in retirement_tokens:
                retired = 1
                break
            cleaned.append(set)

        retirement_flags.append(retired)
        #join all set by a space
        cleaned_scores.append(" ".join(cleaned))

    df["is_retirement"] = retirement_flags
    df["score"] = cleaned_scores
    return df


def extract_tiebreak_from_set(set_str, is_final_set, match_tiebreak_allowed):
    # match tiebreak
    if "(" in set_str and is_final_set and match_tiebreak_allowed:
        tiebreak_to = 10

    # normal tiebreak
    elif "(" in set_str:
        tiebreak_to = 7

    # no tiebreak
    else:
        return "none"

    # calculate tiebreak score
    tb_loser = int(set_str.split("(")[1].split(")")[0])
    if tb_loser < tiebreak_to-2:
        tb_winner =tiebreak_to
        return f"{tb_winner}-{tb_loser}"
    else:
        tb_winner= tb_loser + 2
        return f"{tb_winner}-{tb_loser}"


def extract_tiebreak_score_string_row(row):
    #make sure score is string
    score_str = row.get("score", "")
    if not isinstance(score_str, str):
        score_str = str(score_str)
    score_str = score_str.strip()

    # no score no tiebreaks
    if score_str == "":
        return "none"

    best_of = row["best_of"]
    tourney = row["tourney_name"].lower()
    year = row["tourney_date"].year
    tourney_level = row["tourney_level"]

    sets = score_str.split()
    tiebreaks = []

    # is it a match tiebreak? (final set and grand slam after 2022)
    def match_tiebreak_allowed():
        if year >= 2022 and tourney_level == "G" :
            return True
        if "australian open" in tourney and 2019 <= year <= 2021:
            return True
        return False

    for i, set_str in enumerate(sets):

        # sets indices are 0,1,2
        is_final_set = (i == best_of-1)

        tb = extract_tiebreak_from_set(
            set_str=set_str,
            is_final_set=is_final_set,
            match_tiebreak_allowed=match_tiebreak_allowed()
        )

        tiebreaks.append(tb)

    return " ".join(tiebreaks)

# wraps the function into a df so that it can be piped properly in the clean
def extract_tiebreak_score_string(df):
    # create tiebreak scores row and calculate all the scores
    df["tiebreak_scores"] = df.apply(extract_tiebreak_score_string_row, axis=1)
    return df



#%%
"""
step 8 : quick cleaning test to get quick overview if anything should be flagged
mass checks for unexpected values in all columns 
"""

def data_health_check(df):
    print("DATA HEALTH CHECK:\n")

    # any high number of missing values (not recorded)
    print("Missing values (top 10):")
    missing_counts = df.isna().sum().sort_values(ascending=False)
    print(missing_counts.head(30))
    print("Rows containing missing values:\n")

    for col, count in missing_counts.items():
        if count > 0:
            print(f"--- Missing in column: {col} ({count} rows) ---")
            # Show the rows and the actual values in that column
            print(df[df[col].isna()][[col]].head(5))
            print("\n")

    print("\n")

    # unexpected values?
    expected_surface = {"Hard", "Clay", "Grass", "Carpet"}
    expected_rounds = {"R128","R64","R32","R16","QF","SF","F","RR"}
    expected_levels = {"A","M","G","F"}

    print("Unexpected surface values:")
    print(set(df["surface"].dropna()) - expected_surface)
    print("\n")

    print("Unexpected round values:")
    print(set(df["round"].dropna()) - expected_rounds)
    print("\n")

    print("Unexpected tourney_level values:")
    print(set(df["tourney_level"].dropna()) - expected_levels)
    print("\n")

    # value range checks
    print("Height range (p1/p2):")
    print(df["p1_ht"].min(), "to", df["p1_ht"].max())
    print(df["p2_ht"].min(), "to", df["p2_ht"].max())
    print("\n")

    print("Age range (p1/p2):")
    print(df["p1_age"].min(), "to", df["p1_age"].max())
    print(df["p2_age"].min(), "to", df["p2_age"].max())
    print("\n")

    print("Minutes range:")
    print(df["minutes"].min(), "to", df["minutes"].max())
    print("\n")

    # tournament summary (new tournaments / old)
    print("Tournaments in this dataset:")
    print(sorted(df["tourney_name"].unique()))
    print("\nEND OF REPORT")


#%%
"""
step 7 : clean and save
"""

def clean_matches(df):
    df = (
       df.pipe(normalise_headers)
        .pipe(remove_non_matches)
        .pipe(drop_scoreless_retirements)
        .pipe(filter_atp_events)
        .pipe(filter_matches)
        .pipe(filter_rr_events)
        .pipe(normalise_missing_values)
        .pipe(parse_dates)
        .pipe(convert_numeric)
        .pipe(clean_ranks_and_seeds)
        .pipe(map_hands)
        .pipe(normalize_tourney_names)
        .pipe(clean_categories)
        .pipe(validate_physical_stats)
        .pipe(validate_surface)
        .pipe(validate_round)
        .pipe(validate_best_of)
        .pipe(validate_minutes)
        .pipe(remove_retirement_and_flag)
        .pipe(extract_tiebreak_score_string)
    )
    return df

#%%
"""
step 8 : run df through the mirror row and convert to p1 p2 
makes the df model suitable and prevent bias
"""

from src.data_preprocessing.mirror_rows import mirror_rows
from src.data_preprocessing.convert_to_p1p2 import convert_to_p1p2

def convert_for_model(df):
    df = (
        df.pipe(mirror_rows)
        .pipe(convert_to_p1p2)
     )
    return df

#%%
"""
step 9 : handel the na's
in order for feature calculation 
"""
from src.data_preprocessing.na_handling import handle_na

def handle_missing_values(df):
    return df.pipe(handle_na)

#%%
"""
process year : 
function that takes a year and find the appropriate year in the raw matches files
and puts it through the pipeline
"""

def process_year(year):

    input_path = RAW_MATCHES_PATH / f"atp_matches_{year}.csv"
    output_path = MATCHES_PATH / f"atp_matches_{year}_clean.csv"

    #cleaning pipeline
    df = pd.read_csv(input_path)
    raw_count = len(df)
    df = clean_matches(df)
    df = convert_for_model(df)
    df = handle_missing_values(df)
    data_health_check(df)

    #how many roles removed
    clean_count = len(df)

    net_change = clean_count - raw_count
    print(f"Raw rows: {raw_count}")
    print(f"Final rows after cleaning + mirroring: {clean_count}")
    print(f"Net change: {net_change:+} rows")


    MATCHES_PATH.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f" Cleaned {year} — {len(df)} matches")

#%%
"""
allows years to be cleaned quickly 
"""
if __name__ == "__main__":
    for i in range(2000,2024):
        process_year(i)


