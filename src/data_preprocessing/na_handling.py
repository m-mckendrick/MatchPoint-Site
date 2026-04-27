import pandas as pd
import numpy as np

# with 0
def fill_missing_serve_stats(df):

    serve_cols = [
        "p1_ace","p1_df","p1_svpt","p1_1stin","p1_1stwon","p1_2ndwon","p1_svgms","p1_bpsaved","p1_bpfaced",
        "p2_ace","p2_df","p2_svpt","p2_1stin","p2_1stwon","p2_2ndwon","p2_svgms","p2_bpsaved","p2_bpfaced",
    ]

    for col in serve_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


#fill with ""
#found out the only missing scores were due to retirement before game had been completed so score was NaN
def fill_missing_score(df):
    df["score"] = df["score"].fillna("0-0")
    return df


#if previous excist use that otherwise median
def fill_missing_height(df):

    median_ht = 185

    for prefix in ["p1_", "p2_"]:
        ht_col = prefix + "ht"
        id_col = prefix + "id"

        if ht_col not in df.columns or id_col not in df.columns:
            continue

        # first find last know height for player of certain id
        df[ht_col] = (
            df.groupby(id_col)[ht_col]
            .apply(lambda s: s.ffill().bfill())
            .reset_index(level=0, drop=True)
        )

        # if still na fill with median
        df[ht_col] = df[ht_col].fillna(median_ht)

    return df


def fill_missing_hand(df):

    for prefix in ["p1_", "p2_"]:
        hand_col = prefix + "hand"
        id_col = prefix + "id"

        if hand_col not in df.columns or id_col not in df.columns:
            continue

        # forward/backward fill per player
        df[hand_col] = (
            df.groupby(id_col)[hand_col]
            .apply(lambda s: s.ffill().bfill())
            .reset_index(level=0, drop=True)
        )

        # default is right
        df[hand_col] = df[hand_col].fillna("Right")

    return df


#remove columns that generally arnt used
def drop_unused_columns(df):

    drop_cols = [
        "p1_seed", "p2_seed",
        "p1_entry", "p2_entry",
    ]

    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df

# fill rank with low rank and rank points with 0
# missing rank tends to be a very low unranked player
#very small chance some player are rank but ranking points cant be found dont need to worry about this too much
#rank is the primary strength indicator not rank points
def fill_missing_rank(df):

    if "p1_rank" in df.columns:
        df["p1_rank"] = df["p1_rank"].fillna(500)
    if "p2_rank" in df.columns:
        df["p2_rank"] = df["p2_rank"].fillna(500)

    if "p1_rank_points" in df.columns:
        df["p1_rank_points"] = df["p1_rank_points"].fillna(0)
    if "p2_rank_points" in df.columns:
        df["p2_rank_points"] = df["p2_rank_points"].fillna(0)

    return df


def fill_missing_minutes(df):

    # retirement == 0
    if "is_retirement" in df.columns:
        df.loc[df["minutes"].isna() & (df["is_retirement"] == 1), "minutes"] = 0

    # median fallback
    global_med = df["minutes"].median()
    df["minutes"] = df["minutes"].fillna(global_med)

    return df


def handle_na(df):

    df = fill_missing_score(df)
    df = fill_missing_serve_stats(df)
    df = fill_missing_rank(df)
    df = fill_missing_height(df)
    df = drop_unused_columns(df)
    df = fill_missing_minutes(df)
    df = fill_missing_hand(df)

    return df