import pandas as pd
from pathlib import Path

#same as clean pipeline just reads ranking files
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

def combine_rankings():
    files = [
        RAW_DIR / "atp_rankings_00s.csv",
        RAW_DIR / "atp_rankings_10s.csv",
        RAW_DIR / "atp_rankings_20s.csv",
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

    PROCESSED_DIR.mkdir(exist_ok=True)
    rankings.to_csv(PROCESSED_DIR / "atp_rankings_2000-2020.csv", index=False)

    return rankings

if __name__ == "__main__":
    combine_rankings()