#Data Cleaning and Validation

#settings
import pandas as pd
import numpy as np
from pathlib import Path

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)

RAW_PATH = "data/raw/atp_matches_2024.csv"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

#%%
#step 1 : Normalise Headers and Missing Values
def normalise_headers(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace("-", "_")
    return df

def normalise_missing_values(df):
    #with NA
    missing_markers = ["", " ", ".", "--", "NR", "?", "n/a", "N/A"]
    df = df.replace(missing_markers, pd.NA)
    return df

#%%
#step 2 : Type Conversion

#tourney date int -> datetime
def parse_dates(df):
    df["tourney_date"] = pd.to_datetime( df["tourney_date"], format="%Y%m%d", errors="coerce" )
    return df

#serve stats -> numeric
def convert_numeric(df):
    serve_cols = [c for c in df.columns if c.startswith(("w_", "l_"))]
    df[serve_cols] = df[serve_cols].apply(pd.to_numeric, errors="coerce")
    return df

#all rankings and seedings -> integers
def clean_ranks_and_seeds(df):
    seed_cols = ["winner_seed", "loser_seed"]
    rank_cols = ["winner_rank", "loser_rank", "winner_rank_points", "loser_rank_points"]

    df[seed_cols + rank_cols] = df[seed_cols + rank_cols].replace("", pd.NA)

    df[seed_cols] = df[seed_cols].astype("Int64")
    df[rank_cols] = df[rank_cols].astype("Int64")
    return df



#%%
#step 3 : Value Mapping (help model)
def map_hands(df):
    hand_map = {"R": "Right", "L": "Left", "U": "Unknown"}
    df["winner_hand"] = df["winner_hand"].map(hand_map)
    df["loser_hand"] = df["loser_hand"].map(hand_map)
    return df

#%%
#step 4 : category normalisation
def clean_categories(df):
    df["surface"] = df["surface"].str.title()
    df["round"] = df["round"].str.upper()
    return df

#%%
#step 5 : physical value validation (no crazy values)
def validate_physical_stats(df):
    df = df[df["winner_ht"].between(140, 220) | df["winner_ht"].isna()]
    df = df[df["loser_ht"].between(140, 220) | df["loser_ht"].isna()]
    df = df[df["winner_age"].between(14, 45) | df["winner_age"].isna()]
    df = df[df["loser_age"].between(14, 45) | df["loser_age"].isna()]
    return df


def validate_surface(df):
    valid = {"Hard", "Clay", "Grass"}
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



#%%
#step 6 : remove all matches that are not atp events
def filter_atp_events(df):
    allowed_levels = {"A", "G", "M", "F"}
    df = df[df["tourney_level"].isin(allowed_levels)]

    banned_keywords = [
        "United Cup",
        "Laver Cup",
        "Next Gen Finals",
        "Olympics",
        "Davis Cup",
        "Exhibition"
    ]

    pattern = "|".join(banned_keywords)
    df = df[~df["tourney_name"].str.contains(pattern, case=False, na=False)]

    return df

#%%
#step 7 : add tiebreak scores to data

def extract_tiebreak_from_set(set_str, is_final_set, match_tiebreak_allowed):
    #match tiebreak
    if "(" in set_str and is_final_set and match_tiebreak_allowed:
        tiebreak_to = 10

    #normal tiebreak
    elif "(" in set_str:
        tiebreak_to = 7

    #no tiebreak
    else:
        return "none"

    #calculate tiebreak score
    tb_loser = int(set_str.split("(")[1].split(")")[0])
    if tb_loser < tiebreak_to-2:
        tb_winner =tiebreak_to
        return f"{tb_winner}-{tb_loser}"
    else:
        tb_winner= tb_loser + 2
        return f"{tb_winner}-{tb_loser}"


def extract_tiebreak_score_string_row(row):
    score_str = row["score"]
    best_of = row["best_of"]
    tourney = row["tourney_name"].lower()
    year = row["tourney_date"].year
    tourney_level = row["tourney_level"]

    sets = score_str.split()
    tiebreaks = []

    #is it a match tiebreak? (final set and grand slam after 2022)
    def match_tiebreak_allowed():
        if year >= 2022 and tourney_level == "G" :
            return True
        if "australian open" in tourney and 2019 <= year <= 2021:
            return True
        return False

    for i, set_str in enumerate(sets):
        #sets indices are 0,1,2
        is_final_set = (i == best_of-1)

        tb = extract_tiebreak_from_set(
            set_str=set_str,
            is_final_set=is_final_set,
            match_tiebreak_allowed=match_tiebreak_allowed()
        )

        tiebreaks.append(tb)

    return " ".join(tiebreaks)

#wraps the function into a df so that it can be piped properly in the clean
def extract_tiebreak_score_string(df):
    #create tiebreak scores row and calculate all the scores
    df["tiebreak_scores"] = df.apply(extract_tiebreak_score_string_row, axis=1)
    return df



#%%
#step 8 : quick cleaning test to get quick overview if anything should be flagged
def data_health_check(df):
    print("DATA HEALTH CHECK:\n")

    # any high number of missing values (not recorded)
    print("Missing values (top 10):")
    print(df.isna().sum().sort_values(ascending=False).head(10))
    print("\n")

    # Unexpected values?
    expected_surface = {"Hard", "Clay", "Grass"}
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
    print("Height range (winner/loser):")
    print(df["winner_ht"].min(), "to", df["winner_ht"].max())
    print(df["loser_ht"].min(), "to", df["loser_ht"].max())
    print("\n")

    print("Age range (winner/loser):")
    print(df["winner_age"].min(), "to", df["winner_age"].max())
    print(df["loser_age"].min(), "to", df["loser_age"].max())
    print("\n")

    print("Minutes range:")
    print(df["minutes"].min(), "to", df["minutes"].max())
    print("\n")

    # Tournament summary (new tournaments / old)
    print("Tournaments in this dataset:")
    print(sorted(df["tourney_name"].unique()))
    print("\nEND OF REPORT")


#%%
#final step clean and save

from src.data_preprocessing.mirror_raw_rows import mirror_raw_rows
from src.data_preprocessing.convert_to_p1p2 import convert_to_p1p2

def clean_matches(df):
    df = (
        df.pipe(mirror_raw_rows)

        .pipe(normalise_headers)
        .pipe(filter_atp_events)
        .pipe(normalise_missing_values)
        .pipe(parse_dates)
        .pipe(convert_numeric)
        .pipe(clean_ranks_and_seeds)
        .pipe(map_hands)
        .pipe(clean_categories)
        .pipe(validate_physical_stats)
        .pipe(validate_surface)
        .pipe(validate_round)
        .pipe(validate_best_of)
        .pipe(validate_minutes)
        .pipe(extract_tiebreak_score_string)
        .pipe(convert_to_p1p2)
    )
    return df

#%%
# process year (function that reads raw csv and cleans them all and save them )


def process_year(year):

    input_path = RAW_DIR / f"atp_matches_{year}.csv"
    output_path = PROCESSED_DIR / f"atp_matches_{year}_clean.csv"


    df = pd.read_csv(input_path)
    df_clean = clean_matches(df)
    df_clean = filter_atp_events(df_clean)
    df_clean.to_csv(output_path, index=False)

    print(f" Cleaned {year} — {len(df_clean)} matches")
    #data_health_check(df_clean)

#%%
#if file directly ran cleans and saves 2024
if __name__ == "__main__":
    df = pd.read_csv(RAW_PATH)
    df_clean = clean_matches(df)
    #data_health_check(df_clean)
    df_clean.to_csv(PROCESSED_DIR / "atp_matches_2024_clean.csv", index=False)


