import numpy as np
import pandas as pd


def add_fatigue_features(df, tau_pre=5, tau_in=2, min_effect=0.1):

    df = df.sort_values(["tourney_date", "round_encoded"]).copy()

    #pre tourny fatigue
    def compute_pre_fatigue(player_matches):
        dates = player_matches["tourney_date"].tolist()
        minutes = player_matches["minutes"].tolist()
        tourneys = player_matches["tourney_signature"].tolist()

        fatigue_vals = []

        for current in range(len(player_matches)):
            cumulative = 0

            for prev in range(current):

                # only count matches from previous tournaments
                if tourneys[current] == tourneys[prev]:
                    continue

                days_rest = (dates[current] - dates[prev]).days
                half_weeks_rest = days_rest / 3.5

                decay = np.exp(-half_weeks_rest / tau_pre)
                contrib = minutes[prev] * decay

                if contrib < min_effect:
                    break

                cumulative += contrib

            fatigue_vals.append(cumulative)

        return fatigue_vals

    #in tourny fatigue
    def compute_in_fatigue(player_matches):
        rounds = player_matches["round_encoded"].tolist()
        minutes = player_matches["minutes"].tolist()
        tourneys = player_matches["tourney_signature"].tolist()

        fatigue_vals = []

        for current in range(len(player_matches)):
            cumulative = 0

            for prev in range(current):

                # Only count matches in the SAME tournament
                if tourneys[current] != tourneys[prev]:
                    continue

                round_gap = rounds[current] - rounds[prev]

                decay = np.exp(-round_gap / tau_in)
                contrib = minutes[prev] * decay

                if contrib < min_effect:
                    break

                cumulative += contrib

            fatigue_vals.append(cumulative)

        return fatigue_vals

    # apply to each player
    def compute_for_player(df, player_id_col):
        pre = pd.Series(0.0, index=df.index)
        intra = pd.Series(0.0, index=df.index)

        for _, pm in df.groupby(player_id_col):
            pm = pm.sort_values(["tourney_date", "round_encoded"])

            pre_vals = compute_pre_fatigue(pm)
            in_vals = compute_in_fatigue(pm)

            pre.loc[pm.index] = pre_vals
            intra.loc[pm.index] = in_vals

        return pre, intra

    # compute raw values
    df["p1_fatigue_pre_raw"], df["p1_fatigue_in_raw"] = compute_for_player(df, "p1_id")
    df["p2_fatigue_pre_raw"], df["p2_fatigue_in_raw"] = compute_for_player(df, "p2_id")

    # normalise 0-1
    def normalise(a, b):
        max_val = max(df[a].max(), df[b].max(), 1)
        return df[a] / max_val, df[b] / max_val

    df["p1_fatigue_pre"], df["p2_fatigue_pre"] = normalise(
        "p1_fatigue_pre_raw", "p2_fatigue_pre_raw"
    )

    df["p1_fatigue_in"], df["p2_fatigue_in"] = normalise(
        "p1_fatigue_in_raw", "p2_fatigue_in_raw"
    )

    df.drop(columns=[
        "p1_fatigue_pre_raw", "p1_fatigue_in_raw",
        "p2_fatigue_pre_raw", "p2_fatigue_in_raw"
    ], inplace=True)

    return df
