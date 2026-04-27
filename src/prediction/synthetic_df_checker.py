import os
import pandas as pd


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _save_csv(df: pd.DataFrame, path: str, name: str):
    if df is None:
        return
    full = os.path.join(path, name)
    df.to_csv(full, index=False)


def _compute_player_winrate(history: pd.DataFrame, player_name: str, window: int = 5):
    if history.empty:
        return None

    player_hist = history[
        (history["p1_name"] == player_name) | (history["p2_name"] == player_name)
        ].sort_values(["tourney_date", "match_num"])

    if len(player_hist) < window:
        return None

    last = player_hist.tail(window)
    results = []
    for _, row in last.iterrows():
        if row["p1_name"] == player_name:
            results.append(1 if row["label"] == 1 else 0)
        else:
            results.append(1 if row["label"] == 0 else 0)

    return sum(results) / float(window)


def check_synthetic_df(
        history_all: pd.DataFrame,
        history_same: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        synthetic_features: pd.DataFrame,
        row_full: pd.DataFrame,
        row_model: pd.DataFrame,
        p1_name: str,
        p2_name: str,
        surface: str,
        save_dir: str = None,
):
    print("SYNTHETIC DF CHECKER")
    print("\n")

    if save_dir:
        _ensure_dir(save_dir)
        _save_csv(history_all, save_dir, "history_all.csv")
        _save_csv(history_same, save_dir, "history_same.csv")
        _save_csv(synthetic_df, save_dir, "synthetic_df.csv")
        _save_csv(synthetic_features, save_dir, "synthetic_features.csv")
        _save_csv(row_full, save_dir, "synthetic_row_full.csv")
        _save_csv(row_model, save_dir, "synthetic_row_model.csv")
        print(f"Saved all diagnostic CSVs to: {save_dir}\n")


    print("HISTORY SUMMARY (ALL SURFACES)")
    print(f"Rows: {len(history_all)}")
    print(
        history_all[
            ["tourney_id", "tourney_date", "p1_name", "p2_name"]
        ].tail(10)
    )

    print("\nHISTORY SUMMARY (SAME SURFACE)")
    print(f"Rows: {len(history_same)}")
    print(
        history_same[
            ["tourney_id", "tourney_date", "p1_name", "p2_name"]
        ].tail(10)
    )

    #dupe check
    print("\nDUPLICATE CHECK")
    dupes = synthetic_df.duplicated(subset=["tourney_id", "match_num"]).sum()
    print(f"Duplicate matches in synthetic_df: {dupes}")

    # form feature check
    print("\nFORM FEATURE CHECK")
    syn = row_full.iloc[0]
    form_cols = [c for c in syn.index if "form_" in c]
    non_zero = (syn[form_cols] != 0).sum()
    print(f"Total form features: {len(form_cols)}")
    print(f"Non-zero form features: {non_zero}")

    # check form values are correct
    print("\nFORM CORRECTNESS VALIDATION")

    #p1 all surfaces
    p1_expected_all = _compute_player_winrate(history_all, p1_name, window=5)
    p1_actual_all = syn.get("p1_form_all_5_winrate", None)

    if p1_expected_all is not None and p1_actual_all is not None:
        print(f"P1 all-surface winrate expected: {p1_expected_all:.4f}")
        print(f"P1 all-surface winrate actual:   {p1_actual_all:.4f}")
    else:
        print("P1 all-surface winrate: not enough data or feature missing")

    # p2 all surfaces
    p2_expected_all = _compute_player_winrate(history_all, p2_name, window=5)
    p2_actual_all = syn.get("p2_form_all_5_winrate", None)

    if p2_expected_all is not None and p2_actual_all is not None:
        print(f"\nP2 all-surface winrate expected: {p2_expected_all:.4f}")
        print(f"P2 all-surface winrate actual:   {p2_actual_all:.4f}")
    else:
        print("P2 all-surface winrate: not enough data or feature missing")

    # p1 same surface
    p1_expected_same = _compute_player_winrate(history_same, p1_name, window=5)
    p1_actual_same = syn.get("p1_form_same_5_winrate", None)

    if p1_expected_same is not None and p1_actual_same is not None:
        print(f"\nP1 same-surface winrate expected: {p1_expected_same:.4f}")
        print(f"P1 same-surface winrate actual:   {p1_actual_same:.4f}")
    else:
        print("P1 same-surface winrate: not enough data or feature missing")

    # p2 same surface
    p2_expected_same = _compute_player_winrate(history_same, p2_name, window=5)
    p2_actual_same = syn.get("p2_form_same_5_winrate", None)

    if p2_expected_same is not None and p2_actual_same is not None:
        print(f"\nP2 same-surface winrate expected: {p2_expected_same:.4f}")
        print(f"P2 same-surface winrate actual:   {p2_actual_same:.4f}")
    else:
        print("P2 same-surface winrate: not enough data or feature missing")

    # 5. MOST RECENT MATCH CHECK
    print("\nMOST RECENT MATCH CHECK")

    def most_recent(history: pd.DataFrame, player: str):
        rows = history[
            (history["p1_name"] == player) | (history["p2_name"] == player)
            ]
        if rows.empty:
            return None
        return rows.sort_values(["tourney_date", "match_num"]).iloc[-1]

    p1_last = most_recent(history_all, p1_name)
    p2_last = most_recent(history_all, p2_name)

    if p1_last is not None:
        print("P1 last match:")
        print(p1_last[["tourney_id", "tourney_date", "surface"]])
    else:
        print("P1 last match: none found")

    if p2_last is not None:
        print("\nP2 last match:")
        print(p2_last[["tourney_id", "tourney_date", "surface"]])
    else:
        print("P2 last match: none found")

    print("\n===== CHECK COMPLETE =====\n")
