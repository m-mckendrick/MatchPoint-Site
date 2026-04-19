import numpy as np
import pandas as pd


"""
fatigue index
calculated exponentially in relation to the amount of rest time
then when rest is low, length of match and proximity of matches to each other
build the fatigue value
"""

def add_fatigue_index(df, tau=3, min_effect=0.5):

    df = df.sort_values(["tourney_date", "match_num"]).copy()

    #fatigue calc for 1 player
    def compute_fatigue(player_id_col):
        fatigue_values = []
        for player_id, player_matches in df.groupby(player_id_col):
            dates = player_matches["tourney_date"].tolist()
            minutes = player_matches["minutes"].tolist()

            player_fatigue = []
            #number of matches
            for current_match in range(len(player_matches)):
                cumulative_fatigue = 0
                for previous_match in range(current_match):
                    # how many days since last match
                    days_rest = (dates[current_match] - dates[previous_match]).days
                    # if rest is high decay = 0
                    # if rest is low decay = 1
                    # effective decay of fatigue based on rest
                    decay = np.exp(-days_rest / tau) # tau controls how quickly fatigue fades with rest.

                    # effective fatigue contribution
                    match_fatigue = minutes[previous_match] * decay

                    #break when the effect get so insignificant
                    if match_fatigue < min_effect:
                        break

                    cumulative_fatigue += match_fatigue
                player_fatigue.append(cumulative_fatigue)

            fatigue_values.extend(player_fatigue)

        return pd.Series(fatigue_values, index=df.index)

    # compute raw fatigue
    df["p1_fatigue_raw"] = compute_fatigue("p1_id")
    df["p2_fatigue_raw"] = compute_fatigue("p2_id")

    # matches with no history fill fatigue with 0
    df["p1_fatigue_raw"] = df["p1_fatigue_raw"].fillna(0)
    df["p2_fatigue_raw"] = df["p2_fatigue_raw"].fillna(0)

    # normalise 0–1
    # find the maximum fatigue value if all lower than one set to 1
    max_val = max(df["p1_fatigue_raw"].max(),
                  df["p2_fatigue_raw"].max(), 1)

    df["p1_fatigue_index"] = df["p1_fatigue_raw"] / max_val
    df["p2_fatigue_index"] = df["p2_fatigue_raw"] / max_val

    return df
