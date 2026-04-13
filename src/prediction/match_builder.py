import joblib
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from src.feature_engineering.match_context import encode_round, add_pre_match_context_features

from data.loader import (
    load_matches,
    load_rankings,
    load_processed_matches,
    load_model
)

from src.feature_engineering.feature_builder import build_features
from src.model_training.logistical_regression import prepare_dataset

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
        "name": player_name,
        "rank": ranking_info["rank"],
        "rank_points": ranking_info["rank_points"],
        "age": latest["p1_age"] if is_p1 else latest["p2_age"],
        "height": latest["p1_ht"] if is_p1 else latest["p2_ht"],
        "hand": latest["p1_hand"] if is_p1 else latest["p2_hand"],
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
        "tourney_name": row["tourney_name"],
        "tourney_date":row["tourney_date"],
        "surface": row["surface_raw"],
        "best_of": row["best_of"],
        "tourney_level": row["tourney_level"],
        "draw_size": row["draw_size"] if "draw_size" in rows.columns else None,
        "round" : None
    }

    return meta



#creates match row with all the features being used to predict inside
#keep in mind the player profiles are completely made from just the most recent match
def build_match(df, rankings_df, p1_name, p2_name, tournament, year, round_label, model_features):
    #get tournament metadata
    meta = get_tournament_metadata(df, tournament, year)

    #calculate the cutoff date from matches to be used
    cutoff_date = pd.to_datetime(meta["tourney_date"])

    #create the 2 player profile to add to the match
    p1 = build_player_profile(df, rankings_df, p1_name, cutoff_date)
    p2 = build_player_profile(df, rankings_df, p2_name, cutoff_date)

    #build match row with same columns to the processed matches
    raw = pd.DataFrame([{
        # tournament metadata
        "tourney_id": None,
        "tourney_name": meta["tourney_name"],
        "tourney_date": meta["tourney_date"],
        "surface": meta["surface"],
        "draw_size": meta["draw_size"],
        "tourney_level": meta["tourney_level"],
        "best_of": meta["best_of"],
        "round": round_label,

        #post match data
        "match_num": None,
        "score": None,
        "minutes": None,
        "tiebreak_scores": None,
        "perspective": None,
        "label": None,

        # player 1
        "p1_id": None,
        "p1_name": p1["name"],
        "p1_rank": p1["rank"],
        "p1_rank_points": p1["rank_points"],
        "p1_age": p1["age"],
        "p1_ht": p1["height"],
        "p1_hand": p1["hand"],
        "p1_ioc": None,
        "p1_seed": None,
        "p1_entry": None,

        #p1 post match stats
        "p1_ace": None,
        "p1_df": None,
        "p1_svpt": None,
        "p1_1stin": None,
        "p1_1stwon": None,
        "p1_2ndwon": None,
        "p1_svgms": None,
        "p1_bpsaved": None,
        "p1_bpfaced": None,

        # player 2
        "p2_name": p2["name"],
        "p2_rank": p2["rank"],
        "p2_rank_points": p2["rank_points"],
        "p2_age": p2["age"],
        "p2_ht": p2["height"],
        "p2_hand": p2["hand"],
        "p2_ioc": None,
        "p2_seed": None,
        "p2_entry": None,

        #p2 post match stats
        "p2_ace": None,
        "p2_df": None,
        "p2_svpt": None,
        "p2_1stin": None,
        "p2_1stwon": None,
        "p2_2ndwon": None,
        "p2_svgms": None,
        "p2_bpsaved": None,
        "p2_bpfaced": None,

    }])

    # apply pre-match feature engineering
    #and is a row containing all the matches metadata
    row_full = build_features(raw, df)

    # align with model features (what the model has been trained on)
    #basically the row above but only containing the columns the model will be trained on
    row_model = row_full.reindex(columns=model_features, fill_value=0)

    return row_model, row_full


def main():
    print("Loading model...")
    model, scaler= load_model()

    print("Loading data...")
    df = load_matches()
    rankings_df = load_rankings()

    # prepare dataset once to get the correct feature list
    x, y = prepare_dataset(df)
    model_features = x.columns.tolist()

    print("\nEnter Player 1 name:")
    p1 = input().strip()

    print("Enter Player 2 name:")
    p2 = input().strip()

    print("Enter tournament name (e.g., Wimbledon):")
    tournament = input().strip()

    print("Enter tournament year:")
    year = int(input().strip())

    print("Enter Round (e.g R16, QF, SF, F)")
    round_label = input().strip()

    print("\nBuilding feature row...")
    #row = build_match(df, p1, p2, tournament, year, round_label, model_features)
    row = build_match(df, rankings_df, "Novak Djokovic","Daniil Medvedev","US Open",2018,"F",model_features)

    print("Scaling features...")
    row_scaled = scaler.transform(row)

    print("Predicting...")
    prob = model.predict_proba(row_scaled)[0][1]

    print("\nMATCH PREDICTION")
    print(f"{p1} win probability: {prob*100:.2f}%")
    print(f"{p2} win probability: {(1-prob)*100:.2f}%")

if __name__ == "__main__":
    main()
