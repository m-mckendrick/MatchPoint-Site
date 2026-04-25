
import pandas as pd

from data.loader import (
    load_features,
    load_rankings,
    load_model,
)
from src.feature_engineering.feature_builder import build_features, FEATURE_CONFIG
from src.model_training.logistical_regression import prepare_dataset
from src.prediction.synthetic_df_checker import check_synthetic_df



#profiles
def get_player_ranking(rankings_df, player_id, date):
    rows = rankings_df[
        (rankings_df["player_id"] == player_id)
        & (rankings_df["ranking_date"] <= date)
        ]
    if rows.empty:
        return None

    latest = rows.sort_values("ranking_date", ascending=False).iloc[0]
    return {
        "rank": int(latest["rank"]),
        "rank_points": int(latest["points"]),
    }


def build_player_profile(df, rankings_df, player_name, cutoff_date):
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], errors="coerce")
    cutoff_date = pd.to_datetime(cutoff_date, errors="coerce")

    player_rows = df[
        ((df["p1_name"] == player_name) | (df["p2_name"] == player_name))
        & (df["tourney_date"] <= cutoff_date)
        ]
    if player_rows.empty:
        raise ValueError(f"Player '{player_name}' not found in dataset")

    latest = player_rows.sort_values("tourney_date", ascending=False).iloc[0]
    is_p1 = latest["p1_name"] == player_name
    player_id = latest["p1_id"] if is_p1 else latest["p2_id"]

    ranking_info = get_player_ranking(rankings_df, player_id, cutoff_date)
    if ranking_info is None:
        raise ValueError(f"No ranking available for {player_name} before {cutoff_date}")

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



# tournament metadata
def get_tournament_metadata(df, tournament_name, year):
    rows = df[
        (df["tourney_name"].str.lower() == tournament_name.lower())
        & (df["match_year"] == year)
        ]
    if rows.empty:
        raise ValueError(f"Tournament '{tournament_name}' not found in dataset")

    row = rows.iloc[0]
    tourney_date = pd.to_datetime(row["tourney_date"], errors="coerce")

    meta = {
        "tourney_id": row["tourney_id"],
        "tourney_name": row["tourney_name"],
        "tourney_date": tourney_date,
        "surface": row["surface_raw"],
        "best_of": row["best_of"],
        "tourney_level": row["tourney_level"],
        "draw_size": row.get("draw_size", None),
        "tourney_country": row.get("tourney_country", None),
        "is_indoor": row.get("is_indoor", None),
        "tourney_prestige": row.get("tourney_prestige", None),
        "tourney_signature": row.get("tourney_signature", None),
    }
    return meta


# synthetic row

def make_synthetic_row(meta, p1, p2, round_label):
    tourney_date = pd.to_datetime(meta["tourney_date"])

    return pd.DataFrame([{
        "tourney_id": meta["tourney_id"],
        "tourney_name": meta["tourney_name"],
        "tourney_date": tourney_date,
        "surface": meta["surface"],
        "best_of": meta["best_of"],
        "tourney_level": meta["tourney_level"],
        "round": round_label,
        "match_year": tourney_date.year,
        "match_num": 999999,
        "draw_size": meta["draw_size"],

        # P1
        "p1_id": p1["id"],
        "p1_name": p1["name"],
        "p1_rank": p1["rank"],
        "p1_rank_points": p1["rank_points"],
        "p1_age": p1["age"],
        "p1_ht": p1["height"],
        "p1_hand": p1["hand"],
        "p1_ioc": p1["ioc"],
        "p1_elo_global": p1["elo_global"],
        "p1_elo_surface": p1["elo_surface"],

        # P2
        "p2_id": p2["id"],
        "p2_name": p2["name"],
        "p2_rank": p2["rank"],
        "p2_rank_points": p2["rank_points"],
        "p2_age": p2["age"],
        "p2_ht": p2["height"],
        "p2_hand": p2["hand"],
        "p2_ioc": p2["ioc"],
        "p2_elo_global": p2["elo_global"],
        "p2_elo_surface": p2["elo_surface"],

        "elo_global_diff": p1["elo_global"] - p2["elo_global"],
        "elo_surface_diff": p1["elo_surface"] - p2["elo_surface"],
    }])


#history building

def normalize_perspective(df, player_name):
    df = df.copy()

    mask = df["p2_name"] == player_name

    # swap columns where needed
    swap_cols = [
        ("p1_name", "p2_name"),
        ("p1_ace", "p2_ace"),
        ("p1_df", "p2_df"),
        ("p1_svpt", "p2_svpt"),
        ("p1_1stin", "p2_1stin"),
        ("p1_1stwon", "p2_1stwon"),
        ("p1_2ndwon", "p2_2ndwon"),
        ("p1_svgms", "p2_svgms"),
        ("p1_bpsaved", "p2_bpsaved"),
        ("p1_bpfaced", "p2_bpfaced"),
        ("p1_first_serve_pct", "p2_first_serve_pct"),
        ("p1_first_serve_won_pct", "p2_first_serve_won_pct"),
        ("p1_second_serve_won_pct", "p2_second_serve_won_pct"),
        ("p1_serve_points_won_pct", "p2_serve_points_won_pct"),
        ("p1_double_fault_rate", "p2_double_fault_rate"),
        ("p1_aces_per_game", "p2_aces_per_game"),
        ("p1_bp_save_pct", "p2_bp_save_pct"),
        ("p1_estimated_hold_pct", "p2_estimated_hold_pct"),
        ("p1_serve_efficiency", "p2_serve_efficiency"),
        ("p1_return_points_won_pct", "p2_return_points_won_pct"),
        ("p1_bp_conversion_pct", "p2_bp_conversion_pct"),
        ("p1_estimated_break_pct", "p2_estimated_break_pct"),
    ]

    for a, b in swap_cols:
        df.loc[mask, [a, b]] = df.loc[mask, [b, a]].values

    # fix label
    df.loc[mask, "label"] = 1 - df.loc[mask, "label"]

    return df


def get_history(df, player_name, cutoff, n=10, surface=None, is_p1=False):
    df = df.copy()

    df["tourney_date"] = pd.to_datetime(df["tourney_date"], errors="coerce")

    df = df[df["tourney_date"] < cutoff]

    # fixed alot of issues by making this
    if is_p1 == False:
        df = df[(df["p2_name"] == player_name)]
    else:
        df = df[(df["p1_name"] == player_name)]

    if surface is not None:
        df = df[df["surface_raw"] == surface]

    rows = df.tail(n).copy()

    # surface issue fix

    # rename
    if "surface_raw" in rows.columns:
        rows = rows.rename(columns={"surface_raw": "surface"})

    # drop surface dummies
    for col in ["surface_Clay", "surface_Grass", "surface_Hard", "surface_Carpet"]:
        if col in rows.columns:
            rows = rows.drop(columns=[col])


    history_rows = rows.sort_values(["tourney_date", "match_num"])
    return history_rows.tail(n).copy()


def strip_for_history(df):
    bad = []

    # pre-match context
    bad += [
        "rank_diff", "rank_points_diff", "is_grand_slam", "is_masters",
        "is_best_of_five", "p1_lefty", "p2_lefty", "lefty_diff",
        "surface_Carpet", "surface_Clay", "surface_Grass", "surface_Hard",
        "round_encoded", "age_diff", "height_diff", "tourney_prestige",
        "round_points", "p1_home", "p2_home", "home_advantage_diff",
        "is_indoor",
    ]

    # fatigue
    bad += [c for c in df.columns if "fatigue" in c]

    # existing form
    bad += [c for c in df.columns if "form_" in c]

    # elo diffs / h2h / tournament form
    bad += [c for c in df.columns if "elo_diff" in c or "h2h" in c or "tournament_form" in c]

    return df.drop(columns=bad, errors="ignore")


def assemble_synthetic_df(history_all, history_same, synthetic_row):
    df = pd.concat(
        [history_all, history_same, synthetic_row],
        ignore_index=True
    )

    # dedupe on true match identity (tourney_id + match_num)
    df = df.drop_duplicates(subset=["tourney_id", "match_num"])
    df = df.sort_values(["tourney_date", "match_num"]).reset_index(drop=True)
    return df


#calll build features
def calculate_model_features(synth_df):
    FEATURE_CONFIG.update({
        "pre_match": True,
        "form": True,
        "surface_form": True,
        "fatigue": False,
        "serve_return": False,
        "match_result": False,
        "pressure": False,
        "elo": False,
        "h2h": False,
        "tournament_form": False,
    })
    return build_features(synth_df)


def extract_prediction_row(features_df, model_features):
    row = features_df.iloc[[-1]].copy()
    return row.reindex(columns=model_features, fill_value=0)


def build_match(
        df,
        rankings_df,
        p1_name,
        p2_name,
        tournament,
        year,
        round_label,
        model_features,
        debug=False,
):

    meta = get_tournament_metadata(df, tournament, year)
    cutoff = meta["tourney_date"]

    p1 = build_player_profile(df, rankings_df, p1_name, cutoff)
    p2 = build_player_profile(df, rankings_df, p2_name, cutoff)

    synthetic_row = make_synthetic_row(meta, p1, p2, round_label)

    # create match histories where p1 of synthetic row is always p1 and the same for p2
    raw_p1_all  = get_history(df, p1_name, cutoff, n=5, surface=None, is_p1=True)
    raw_p2_all  = get_history(df, p2_name, cutoff, n=5, surface=None, is_p1=False)
    raw_p1_same = get_history(df, p1_name, cutoff, n=5, surface=meta["surface"], is_p1=True)
    raw_p2_same = get_history(df, p2_name, cutoff, n=5, surface=meta["surface"], is_p1=False)

    # normalise perspective (swap match perspective)
    # due to the form builder only looking at p1 stat columns need to have both in that format for form to be calculated
    p1_all  = normalize_perspective(raw_p1_all,  p1_name)
    p1_same = normalize_perspective(raw_p1_same, p1_name)

    p2_all  = normalize_perspective(raw_p2_all,  p2_name)
    p2_same = normalize_perspective(raw_p2_same, p2_name)


    history_all  = pd.concat([p1_all,  p2_all],  ignore_index=True)
    history_same = pd.concat([p2_same, p1_same], ignore_index=True)

    #get rid of dup columns (column that will be added)
    history_all  = strip_for_history(history_all)
    history_same = strip_for_history(history_same)

    # build feature ready synthetic df
    synth_df = assemble_synthetic_df(history_all, history_same, synthetic_row)

    features = calculate_model_features(synth_df)
    row_model = extract_prediction_row(features, model_features)
    row_full = features.iloc[[-1]].copy()

    # call checker if u want to
    if debug:
        check_synthetic_df(
            history_all=history_all,
            history_same=history_same,
            synthetic_df=synth_df,
            synthetic_features=features,
            row_full=row_full,
            row_model=row_model,
            p1_name=p1_name,
            p2_name=p2_name,
            surface=meta["surface"],
            save_dir="tests/prediction/synthetic_dfs",
        )

    return row_model, row_full



def main():
    print("Loading model...")
    model, scaler, model_features = load_model("models/xgboost_3.pkl")

    print("Loading data...")
    df = load_features()
    rankings_df = load_rankings()

    # sanity: ensure model_features align with what we compute
    _x, _y = prepare_dataset(df)
    print("columns used:", model_features)

    print("\nBuilding synthetic match: Novak Djokovic vs Daniil Medvedev at Barcelona 2018 (F)")
    row_model, row_full = build_match(
        df,
        rankings_df,
        "Novak Djokovic",
        "Daniil Medvedev",
        "Barcelona",
        2018,
        "F",
        model_features,
        debug=True,
    )

    print("Predicting...")
    prob = model.predict_proba(row_model)[0][1]
    print("\nMATCH PREDICTION")
    print(f" p1 win probability: {prob*100:.2f}%")
    print(f" p2 win probability: {(1-prob)*100:.2f}%")


if __name__ == "__main__":
    main()
