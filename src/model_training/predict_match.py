import joblib
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

MODEL_PATH = Path("models/logreg_baseline.pkl")
DATA_PATH = Path("data/processed")

from src.feature_engineering.feature_builder import build_features
from src.model_training.logistical_regression import prepare_dataset

def load_model():
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"], bundle["scaler"]


def load_data():
    df = build_features(DATA_PATH)
    return df

#For this player, what’s the most recent match in the dataset, and were they stored as p1 or p2 in that match?
def get_player_latest_stats(df, player_name, cutoff_date):
    # find the most recent matches for this player both wins and losses
    player_rows = df[(df["p1_name"] == player_name) | (df["p2_name"] == player_name)]

    # filter to matches BEFORE the cutoff date
    player_rows = player_rows[player_rows["tourney_date"] < cutoff_date]

    if player_rows.empty:
        raise ValueError(f"Player '{player_name}' not found in dataset")

    # sort by date newest first
    latest = player_rows.sort_values("tourney_date", ascending=False).iloc[0]

    # return row in p1 perspective
    if latest["p1_name"] == player_name:
        return latest, "p1"
    else:
        return latest, "p2"

#basically take the latest match each player has played and simulate a match between them
def build_match(df, p1_name, p2_name, model_features, cutoff_date):
    # get latest stats for each player
    p1_row, _ = get_player_latest_stats(df, p1_name, cutoff_date)
    p2_row, _ = get_player_latest_stats(df, p2_name, cutoff_date)

    #in order to get p1 vs p2 format just take first player's match and overwrite the p2 with the other players stats'
    row = p1_row.copy()

    # overwrite p2 stats with the second player’s stats
    for col in df.columns:
        if col.startswith("p2_"):
            matching = col.replace("p2_", "p1_")
            if matching in p2_row:
                row[col] = p2_row[matching]

    # drop label
    row = row[model_features]

    #na handel
    if row.isna().sum().sum() > 0:
        print("Warning: NA found in prediction row")
        print(row.isna().sum()[row.isna().sum() > 0])
    row = row.fillna(0)



    return row.to_frame().T  # return as dataframe


#match prediction atm super biased as only takes last match of the players into account
#match is simulated as if it took place in the 2019-2020 off season
def main():
    print("Loading model...")
    model, scaler= load_model()

    print("Loading data...")
    df = load_data()

    # prepare dataset once to get the correct feature list
    x, y = prepare_dataset(df)
    model_features = x.columns.tolist()

    print("\nEnter Player 1 name:")
    p1 = input().strip()

    print("Enter Player 2 name:")
    p2 = input().strip()

    print("Enter match date (YYYY-MM-DD):")
    date_str = input().strip()
    cutoff_date = pd.to_datetime(date_str)


    print("\nBuilding feature row...")
    row = build_match(df, p1, p2, model_features, cutoff_date)

    print("Scaling features...")
    row_scaled = scaler.transform(row)

    print("Predicting...")
    prob = model.predict_proba(row_scaled)[0][1]

    print("\nMATCH PREDICTION")
    print(f"{p1} win probability: {prob*100:.2f}%")
    print(f"{p2} win probability: {(1-prob)*100:.2f}%")

if __name__ == "__main__":
    main()
