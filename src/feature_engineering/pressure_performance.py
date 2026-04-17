import pandas as pd
import numpy as np

def safe_div(n, d):
    if d == 0 or pd.isna(d):
        return 0.0
    return n / d


#my pressure performance index
def add_pressure_performance_features(df):
    """
    Computes pressure performance features using:
        - break points saved / faced
        - break points converted / created
        - tiebreak performance
        - deciding set performance
        - close set performance
        - combined pressure index
    """

    # break point performance
    # saved is more significant than converted due to expectation to hold serve
    df["p1_bp_save_pct"] = df.apply(
        lambda r: safe_div(r["p1_bpsaved"], r["p1_bpfaced"]), axis=1
    )
    df["p2_bp_save_pct"] = df.apply(
        lambda r: safe_div(r["p2_bpsaved"], r["p2_bpfaced"]), axis=1
    )

    # break points converted
    df["p1_bp_conv_pct"] = df.apply(
        lambda r: safe_div(r["p2_bpsaved"], r["p2_bpfaced"]), axis=1
    )
    df["p2_bp_conv_pct"] = df.apply(
        lambda r: safe_div(r["p1_bpsaved"], r["p1_bpfaced"]), axis=1
    )

    # tiebreak performance
    df["p1_tiebreak_win_pct"] = df.apply(
        lambda r: safe_div(r["p1_tiebreaks_won"], r["p1_tiebreaks_won"] + r["p2_tiebreaks_won"]),
        axis=1
    )
    df["p2_tiebreak_win_pct"] = df.apply(
        lambda r: safe_div(r["p2_tiebreaks_won"], r["p1_tiebreaks_won"] + r["p2_tiebreaks_won"]),
        axis=1
    )

    # deciding set performance
    def deciding_set_win(row):
        score = row["score"]
        best_of = int(row["best_of"])

        sets = score.split()

        # If match did not reach deciding set
        if len(sets) < best_of:
            return 0, 0

        deciding_set = sets[best_of - 1]  # 0-indexed

        # remove tiebreak part
        base = deciding_set.split("(")[0]
        g1, g2 = map(int, base.split("-"))

        if g1 > g2:
            return 1, 0
        else:
            return 0, 1

    deciding = df.apply(deciding_set_win, axis=1, result_type="expand")
    df["p1_deciding_set_win"] = deciding[0]
    df["p2_deciding_set_win"] = deciding[1]

    # performance in close sets
    def close_set_stats(row):
        sets = row["score"].split()
        p1_close = 0
        p2_close = 0

        for s in sets:
            base = s.split("(")[0]
            g1, g2 = map(int, base.split("-"))
            if (g1, g2) in [(7, 5), (7, 6), (6, 4)]:
                p1_close += 1
            if (g2, g1) in [(7, 5), (7, 6), (6, 4)]:
                p2_close += 1

        return p1_close, p2_close

    close = df.apply(close_set_stats, axis=1, result_type="expand")
    df["p1_close_sets_won"] = close[0]
    df["p2_close_sets_won"] = close[1]

    # the combined pressure performance index
    df["p1_pressure_index"] = (
            0.4 * df["p1_bp_save_pct"] +
            0.3 * df["p1_bp_conv_pct"] +
            0.2 * df["p1_tiebreak_win_pct"] +
            0.1 * df["p1_close_sets_won"]
    )

    df["p2_pressure_index"] = (
            0.4 * df["p2_bp_save_pct"] +
            0.3 * df["p2_bp_conv_pct"] +
            0.2 * df["p2_tiebreak_win_pct"] +
            0.1 * df["p2_close_sets_won"]
    )

    # prestige + atp point = importance
    df["match_importance"] = (
            0.6 * df["tourney_prestige"] +
            0.4 * np.log1p(df["round_points"])
    )

    # normalize to 0–1
    df["pressure_multiplier"] = df["match_importance"] / df["match_importance"].max()

    # scale pressure index
    df["p1_pressure_index"] *= df["pressure_multiplier"]
    df["p2_pressure_index"] *= df["pressure_multiplier"]

    # stage_pressure = a measure of teh performance under pressure with respect to the importance of the match.
    df["stage_pressure"] = df["pressure_multiplier"]


    return df
