#simple match context features
import pandas as pd
import numpy as np
import math

# defined here so not redefined every single row
ROUND_TO_PLAYERS = {
    "F": 2,
    "SF": 4,
    "QF": 8,
    "R16": 16,
    "R32": 32,
    "R64": 64,
    "R128": 128
}

#in order to encode the round of a tournament correctly need to use this function
def encode_round(row):
    draw = row["draw_size"]
    rnd = row["round"]

    # players remaining in this round
    players_remaining = ROUND_TO_PLAYERS.get(rnd)

    #cehck for values like RR (round robin)
    if players_remaining is None:
        return None

    # drawsize effectivly rounded up to nearest power of 2 check notes for maths explanation
    first_round_players = 2 ** int(math.ceil(math.log2(draw)))

    #  round calculated be finding the number of times the player have been halved
    # +1 so that round 1 == 1 not 0
    round_number = int(np.log2(first_round_players / players_remaining)) + 1

    return round_number



def add_match_context_features(df: pd.DataFrame) -> pd.DataFrame:

    #rank and more specific rank points
    df["rank_diff"] = df["p2_rank"] - df["p1_rank"]
    df["rank_points_diff"] = df["p2_rank_points"] - df["p1_rank_points"]
    #seed significance (pressure?)
    df["p1_unseeded"] = df["p1_seed"].isna().astype(int)
    df["p2_unseeded"] = df["p2_seed"].isna().astype(int)

    #if both seed calculate difference
    df["seed_diff"] = np.where(
        (df["p1_unseeded"] == 0) & (df["p2_unseeded"] == 0),
        df["p2_seed"] - df["p1_seed"],
        np.nan
    )

    #o/1 colums required for machine learning models
    df["is_grand_slam"] = (df["tourney_level"] == "G").astype(int)
    df["is_masters"] = (df["tourney_level"] == "M").astype(int)
    #usefull as older master and other tournaments had 5 sets (not just slams)
    df["is_best_of_five"] = (df["best_of"] == 5).astype(int)

    #retirement locater
    if "score" in df.columns:
        df["is_retirement"] = df["score"].str.contains("RET", case=False, na=False).astype(int)
    else:
        df["is_retirement"] = 0

    #changes match pattern dynamics
    df["p1_lefty"] = (df["p1_hand"] == "L").astype(int)
    df["p2_lefty"] = (df["p2_hand"] == "L").astype(int)
    df["lefty_diff"] = df["p1_lefty"] - df["p2_lefty"]

    #takes surface columns makes 0/1 colums for each surface and fill them
    df = pd.get_dummies(df, columns=["surface"], prefix="surface")

    #round encoder called
    df["round_encoded"] = df.apply(encode_round, axis=1)

    #can help era context
    df["match_year"] = pd.to_datetime(df["tourney_date"]).dt.year

    #younger usually more durable
    df["age_diff"] = df["p2_age"] - df["p1_age"]

    #heavy influence on serve
    df["height_diff"] = df["p2_ht"] - df["p1_ht"]

    #help tournament understand importance maybe has effect on pressure
    prestige_map = {"G": 4, "M": 3, "A": 2, "B": 1}
    df["tourney_prestige"] = df["tourney_level"].map(prestige_map)

    return df
