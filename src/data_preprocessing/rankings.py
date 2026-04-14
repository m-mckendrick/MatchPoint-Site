import pandas as pd
from pathlib import Path

#same as clean pipeline just reads ranking files
from data.loader import RAW_MATCHES_PATH, PROCESSED_PATH

"""
combine rankings : find ranking files and combines them 
and sort them so that player ranking are shown from earliest to latest
"""
def combine_rankings():
    files = [
        RAW_MATCHES_PATH / "atp_rankings_00s.csv",
        RAW_MATCHES_PATH / "atp_rankings_10s.csv",
        RAW_MATCHES_PATH / "atp_rankings_20s.csv",
        ]

    df_list = []
    #turn all date to datetime format
    for f in files:
        #use excisting headers
        temp = pd.read_csv(f)

        temp = temp.rename(columns={"player": "player_id"})

        #convert date
        temp["ranking_date"] = pd.to_datetime(temp["ranking_date"], format="%Y%m%d")

        df_list.append(temp)


    rankings = pd.concat(df_list, ignore_index=True)
    #sort rankings so player ranking are all shown from earliest to latest
    rankings = rankings.sort_values(["player_id", "ranking_date"])

    PROCESSED_PATH.mkdir(exist_ok=True)
    rankings.to_csv(PROCESSED_PATH / "atp_rankings_2000-2020.csv", index=False)

    return rankings

if __name__ == "__main__":
    combine_rankings()