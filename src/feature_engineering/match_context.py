#simple match context features
import pandas as pd

#in order to encode the round of a tournament correctly need to use this function
def encode_round(row):
    draw = row["draw_size"]
    rnd = row["round"]

    if draw == 128:
        mapping = {"R128": 1, "R64": 2, "R32": 3, "R16": 4, "QF": 5, "SF": 6, "F": 7}
    elif draw == 64:
        mapping = {"R64": 1, "R32": 2, "R16": 3, "QF": 4, "SF": 5, "F": 6}
    elif draw == 32:
        mapping = {"R32": 1, "R16": 2, "QF": 3, "SF": 4, "F": 5}
    else:
        return None

    return mapping.get(rnd, None)


def match_context(df: pd.DataFrame) -> pd.DataFrame:
    #rank and more specific rank points
    df["rank_diff"] = df["winner_rank"] - df["loser_rank"]
    df["rank_points_diff"] = df["winner_rank_points"] - df["loser_rank_points"]
    #expectations in tournament (pressure?)
    df["seed_diff"] = df["winner_seed"].fillna(0) - df["loser_seed"].fillna(0)

    #o/1 colums required for machine learning models
    df["is_grand_slam"] = (df["tourney_level"] == "G").astype(int)
    df["is_masters"] = (df["tourney_level"] == "M").astype(int)
    #usefull as older master and other tournaments had 5 sets (not just slams)
    df["is_best_of_five"] = (df["best_of"] == 5).astype(int)
    #changes match pattern dynamics
    df["is_lefty_vs_righty"] = (df["winner_hand"] != df["loser_hand"]).astype(int)

    #weather context
    df["is_indoor"] = df["surface"].str.contains("Indoor", case=False, na=False).astype(int)

    #takes surface columns makes 0/1 colums for each surface and fill them
    df = pd.get_dummies(df, columns=["surface"], prefix="surface")

    #round encoder called
    df["round_encoded"] = df.apply(encode_round, axis=1)

    #can help era context
    df["match_year"] = pd.to_datetime(df["tourney_date"]).dt.year

    #weather context
    df["is_indoor"] = df["surface"].str.contains("Indoor", case=False, na=False).astype(int)

    #home advantage/reginal patterns (south america clay/us clay)
    df["tourney_country"] = df["tourney_country"]
    df["winner_home"] = (df["winner_ioc"] == df["tourney_country"]).astype(int)
    df["loser_home"] = (df["loser_ioc"] == df["tourney_country"]).astype(int)
    df["home_advantage_diff"] = df["winner_home"] - df["loser_home"]


    #younger usually more durable
    df["age_diff"] = df["winner_age"] - df["loser_age"]

    #heavy influence on serve
    df["height_diff"] = df["winner_ht"] - df["loser_ht"]

    #help tournament understand importance maybe has effect on pressure
    prestige_map = {"G": 4, "M": 3, "A": 2, "B": 1}
    df["tourney_prestige"] = df["tourney_level"].map(prestige_map)

    #fatigue
    df["match_minutes"] = df["minutes"]

    return df



























