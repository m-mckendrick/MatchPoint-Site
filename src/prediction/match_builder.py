import joblib
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from src.feature_engineering.match_context import encode_round, add_pre_match_context_features

from data.loader import (
    load_matches,
    load_rankings,
    load_features,
    load_model
)

from src.feature_engineering.feature_builder import build_features, FEATURE_CONFIG


from src.model_training.logistical_regression import prepare_dataset

import os
from src.prediction.synthetic_df_checker import check_synthetic_df

from tests.conftest import synthetic_row

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


#get time accurate player ranking
def get_player_ranking(rankings_df, player_id, date):
    #get all ranking of player where id is same and ranking date is = or less than the date
    rows = rankings_df[(rankings_df["player_id"] == player_id) & (rankings_df["ranking_date"] <= date)]
    #no rankings
    if rows.empty:
        return None
    #get latest ranks and sort by latest
    latest = rows.sort_values("ranking_date", ascending=False).iloc[0]
    return {
        "rank": latest["rank"],
        "rank_points": latest["points"]
    }

#For this player, what’s the most recent match in the dataset, and were they stored as p1 or p2 in that match?
def build_player_profile(df, rankings_df, player_name, cutoff_date):
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], errors="coerce")
    cutoff_date = pd.to_datetime(cutoff_date, errors="coerce")
    # find the most recent matches for this player both wins and losses
    player_rows = df[
        ((df["p1_name"] == player_name) | (df["p2_name"] == player_name))
        & (df["tourney_date"] <= cutoff_date)
    ]
    if player_rows.empty:
        raise ValueError(f"Player '{player_name}' not found in dataset")

    # most recent match
    latest = player_rows.sort_values("tourney_date", ascending=False).iloc[0]

    is_p1 = latest["p1_name"] == player_name
    player_id= latest["p1_id"] if is_p1 else latest["p2_id"]

    #get ranking with rankings csv
    ranking_info = get_player_ranking(rankings_df, player_id, cutoff_date)
    if ranking_info is None:
        raise ValueError(f"No ranking available for {player_name} before {cutoff_date}")


    #now creates player profile with latest stats
    profile = {
        "id": player_id,
        "name": player_name,
        "rank": ranking_info["rank"],
        "rank_points": ranking_info["rank_points"],
        "elo_global": latest["p1_elo_global"] if is_p1 else latest["p2_elo_global"],
        "elo_surface": latest["p1_elo_surface"] if is_p1 else latest["p2_elo_surface"],
        "age": latest["p1_age"] if is_p1 else latest["p2_age"],
        "height": latest["p1_ht"] if is_p1 else latest["p2_ht"],
        "hand": latest["p1_hand"] if is_p1 else latest["p2_hand"],
        "ioc": latest["p1_ioc"] if is_p1 else latest["p2_ioc"],
    }

    return profile

#look at tournament matches in the year and collect the metadata
def get_tournament_metadata(df, tournament_name, year):
    #filter df to show matches at tournament at the same year
    rows=df[
        (df["tourney_name"].str.lower() == tournament_name.lower())
        & (df["match_year"] == year)
    ]

    if rows.empty:
        raise ValueError(f"Tournament '{tournament_name}' not found in dataset")

    #becasue the metadata is constant throughout the tournament just take first row
    row=rows.iloc[0]

    #tornament date included as it's the date of the tournament starting
    meta = {
        "tourney_id": row["tourney_id"],
        "tourney_name": row["tourney_name"],
        "tourney_date":row["tourney_date"],
        "surface": row["surface_raw"],
        "best_of": row["best_of"],
        "tourney_level": row["tourney_level"],
        "draw_size": row["draw_size"] if "draw_size" in rows.columns else None,
        "round" : None,
        "tourney_country": row.get("tourney_country", None),
        "is_indoor": row.get("is_indoor", None),
        "tourney_prestige": row.get("tourney_prestige", None),
        "tourney_signature": row.get("tourney_signature", None),

        "round_encoded": None,
    }

    return meta

#in a feature building format
def get_recent_matches(df, player_name, cutoff_date, n=10, surface=None):
    # Step 1: filter matches involving player before cutoff
    cutoff_date = pd.to_datetime(cutoff_date, errors="coerce")

    rows = df.loc[
        ((df["p1_name"] == player_name) | (df["p2_name"] == player_name)) &
        (df["tourney_date"] < cutoff_date)
        ]

    if rows.empty:
        return rows

    if surface is not None:
        rows = rows.loc[rows["surface_raw"] == surface]
    if rows.empty:
        return rows

    rows = rows.tail(n).copy()

    # surface issue fix

    # rename
    if "surface_raw" in rows.columns:
        rows = rows.rename(columns={"surface_raw": "surface"})

    # drop surface dummies
    for col in ["surface_Clay", "surface_Grass", "surface_Hard", "surface_Carpet"]:
        if col in rows.columns:
            rows = rows.drop(columns=[col])

    return rows



def strip_prematch_features(df: pd.DataFrame) -> pd.DataFrame:

    # columns created by add_pre_match_context_features
    match_context_cols = [
        "rank_diff",
        "rank_points_diff",
        "is_grand_slam",
        "is_masters",
        "is_best_of_five",
        "p1_lefty",
        "p2_lefty",
        "lefty_diff",
        "surface_raw",
        "surface_Carpet",
        "surface_Clay",
        "surface_Grass",
        "surface_Hard",
        "tourney_signature",
        "round_encoded",
        "match_year",
        "age_diff",
        "height_diff",
        "tourney_prestige",
        "round_points",
        "tourney_country",
        "p1_home",
        "p2_home",
        "home_advantage_diff",
        "is_indoor",
    ]

    # fatigue columns
    fatigue_cols = [
        "p1_fatigue_pre",
        "p2_fatigue_pre",
        "p1_fatigue_in",
        "p2_fatigue_in",
    ]

    form_cols = [c for c in df.columns if "form_" in c]

    cols_to_drop = match_context_cols + fatigue_cols + form_cols

    # Drop only if they exist
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

    return df




#creates match row with all the features being used to predict inside
#keep in mind the player profiles are completely made from just the most recent match
def build_match(df, rankings_df, p1_name, p2_name, tournament, year, round_label, model_features , return_history=False, debug=False):
    #get tournament metadata
    meta = get_tournament_metadata(df, tournament, year)
    print (meta["surface"])
    #calculate the cutoff date from matches to be used
    cutoff_date = meta["tourney_date"]

    #create the 2 player profile to add to the match
    p1 = build_player_profile(df, rankings_df, p1_name, cutoff_date)
    p2 = build_player_profile(df, rankings_df, p2_name, cutoff_date)

    #build match row with same columns to the processed matches
    raw = pd.DataFrame([{
        # tournament metadata
        "tourney_id": meta["tourney_id"],
        "tourney_name": meta["tourney_name"],
        "tourney_date": pd.to_datetime(meta["tourney_date"]),

        "surface": meta["surface"],

        "draw_size": meta["draw_size"],
        "tourney_level": meta["tourney_level"],
        "best_of": meta["best_of"],
        "round": round_label,
        "match_num": 999999,
        "label": 0,
        "is_retirement":0,

        # player 1
        "p1_id": p1["id"],
        "p1_name": p1["name"],
        "p1_rank": int(p1["rank"]),
        "p1_rank_points": int(p1["rank_points"]),
        "p1_age": p1["age"],
        "p1_ht": p1["height"],
        "p1_hand": p1["hand"],
        "p1_ioc": p1["ioc"],
        "p1_elo_global": p1["elo_global"],
        "p1_elo_surface": p1["elo_surface"],

        # player 2
        "p2_id": p2["id"],
        "p2_name": p2["name"],
        "p2_rank": int(p2["rank"]),
        "p2_rank_points": int(p2["rank_points"]),
        "p2_age": p2["age"],
        "p2_ht": p2["height"],
        "p2_hand": p2["hand"],
        "p2_ioc": p2["ioc"],
        "p2_elo_global": p2["elo_global"],
        "p2_elo_surface": p2["elo_surface"],

        "elo_global_diff": p1["elo_global"] - p2["elo_global"],
        "elo_surface_diff": p1["elo_surface"] - p2["elo_surface"],


        # Synthetic match result placeholders (all zeros)
        "p1_sets_won": 0, "p2_sets_won": 0,
        "p1_sets_lost": 0, "p2_sets_lost": 0,
        "p1_games_won": 0, "p2_games_won": 0,
        "p1_games_lost": 0, "p2_games_lost": 0,
        "p1_tiebreaks_won": 0, "p2_tiebreaks_won": 0,
        "p1_deciding_set_win": 0, "p2_deciding_set_win": 0,
        "p1_straight_sets_win": 0, "p2_straight_sets_win": 0,
        "p1_close_sets_won": 0, "p2_close_sets_won": 0,
        "p1_pressure_index": 0, "p2_pressure_index": 0,
        "p1_bp_save_pct": 0, "p2_bp_save_pct": 0,
        "p1_bp_conversion_pct": 0, "p2_bp_conversion_pct": 0,
        "p1_first_serve_pct": 0, "p2_first_serve_pct": 0,
        "p1_first_serve_won_pct": 0, "p2_first_serve_won_pct": 0,
        "p1_second_serve_won_pct": 0, "p2_second_serve_won_pct": 0,
        "p1_double_fault_rate": 0, "p2_double_fault_rate": 0,
        "p1_aces_per_game": 0, "p2_aces_per_game": 0,
        "p1_serve_points_won_pct": 0, "p2_serve_points_won_pct": 0,
        "p1_estimated_hold_pct": 0, "p2_estimated_hold_pct": 0,
        "p1_serve_efficiency": 0, "p2_serve_efficiency": 0,
        "p1_return_points_won_pct": 0, "p2_return_points_won_pct": 0,
        "p1_estimated_break_pct": 0, "p2_estimated_break_pct": 0,

    }])


    # history logic
    p1_hist_all = get_recent_matches(df, p1_name, cutoff_date, n=10)
    p2_hist_all = get_recent_matches(df, p2_name, cutoff_date, n=10)
    surface = meta["surface"]
    print(surface)
    p1_hist_same = get_recent_matches(df, p1_name, cutoff_date, n=10, surface=surface)
    p2_hist_same = get_recent_matches(df, p2_name, cutoff_date, n=10, surface=surface)

    history = pd.concat([p1_hist_all, p1_hist_same, p2_hist_all, p2_hist_same],ignore_index=True)
    # remove mirrored duplicates using TRUE match identity

    history = history.drop_duplicates(subset=["tourney_id", "match_num", "p1_id", "p2_id"])
    # sort only if history is non-empty
    if not history.empty:
        history = history.sort_values(["tourney_date", "match_num"])

    history_striped = strip_prematch_features(history)

    # append synthetic row
    #issue with surface is here
    synthetic_df = pd.concat([history_striped, raw], ignore_index=True)
    synthetic_df = synthetic_df.sort_values(["tourney_date", "match_num"]).reset_index(drop=True)

    FEATURE_CONFIG["pre_match"]= True
    FEATURE_CONFIG["fatigue"] = True

    FEATURE_CONFIG["serve_return"] = False
    FEATURE_CONFIG["match_result"] = False
    FEATURE_CONFIG["pressure"] = False
    FEATURE_CONFIG["elo"] = False

    FEATURE_CONFIG["form"] = True
    FEATURE_CONFIG["surface_form"] = True

    FEATURE_CONFIG["h2h"] = False
    FEATURE_CONFIG["tournament_form"] = False


    # apply pre-match feature engineering
    #and is a row containing all the matches metadata
    synthetic_features = build_features(synthetic_df)

    #this is correct currently
    row_full = synthetic_features.iloc[[-1]].copy()

    # align with model features (what the model has been trained on)
    #basically the row above but only containing the columns the model will be trained on
    row_model = row_full.reindex(columns=model_features, fill_value=0)

    if debug:
        history.to_csv(
            os.path.join(PROJECT_ROOT, "tests", "prediction", "synthetic_dfs", "recent_matches_history.csv"),
            index=False
        )

        history_striped.to_csv(
            os.path.join(PROJECT_ROOT, "tests", "prediction", "synthetic_dfs", "history_striped.csv"),
            index=False
        )

        raw.to_csv(
            os.path.join(PROJECT_ROOT, "tests", "prediction", "synthetic_dfs", "raw_row.csv"),
            index=False
        )

        synthetic_df.to_csv(
            os.path.join(PROJECT_ROOT, "tests", "prediction", "synthetic_dfs", "synthetic_df.csv"),
            index=False
        )


        synthetic_features.to_csv(
            os.path.join(PROJECT_ROOT, "tests", "prediction", "synthetic_dfs", "synthetic_features.csv"),
            index=False
        )

        row_full.to_csv(
            os.path.join(PROJECT_ROOT, "tests", "prediction", "synthetic_dfs", "synthetic_row_full.csv"),
            index=False
        )

        row_model.to_csv(
            os.path.join(PROJECT_ROOT, "tests", "prediction", "synthetic_dfs", "synthetic_row_model.csv"),
            index=False
        )
        check_synthetic_df(
            history=history,
            synthetic_df=synthetic_df,
            synthetic_features=synthetic_features,
            p1_name=p1_name,
            p2_name=p2_name,
            surface=meta["surface"]
        )

    if return_history:
        return row_model, row_full, history_striped

    return row_model, row_full


def main():
    print("Loading model...")
    model, scaler, model_features = load_model("models/xgboost_3.pkl")

    print("Loading data...")
    df = load_features()
    rankings_df = load_rankings()
    print(model_features)
    # prepare dataset once to get the correct feature list
    x, y = prepare_dataset(df)


    print("\nEnter Player 1 name:")
    #p1 = input().strip()

    print("Enter Player 2 name:")
    #p2 = input().strip()

    print("Enter tournament name (e.g., Wimbledon):")
    #tournament = input().strip()

    print("Enter tournament year:")
    #year = int(input().strip())

    print("Enter Round (e.g R16, QF, SF, F)")
    #round_label = input().strip()

    print("\nBuilding feature row...")
    #row = build_match(df, p1, p2, tournament, year, round_label, model_features)
    row_model, row_full = build_match(df, rankings_df, "Novak Djokovic","Daniil Medvedev","Barcelona",2018,"F",model_features,debug=True)

    print("Predicting...")
    prob = model.predict_proba(row_model)[0][1]

    print("\nMATCH PREDICTION")
    print(f" p1 win probability: {prob*100:.2f}%")
    print(f" p2 win probability: {(1-prob)*100:.2f}%")

if __name__ == "__main__":
    main()
