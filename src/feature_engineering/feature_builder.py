import pandas as pd
from pathlib import Path

from .serve_return import add_serve_return_features
from .match_context import add_pre_match_context_features
from .match_context import add_post_match_context_features

def build_features(processed_dir="data/processed/matches"):
    #get all csv that have been cleaned
    #coverts folder path to folder object allows year sorting etc.
    processed_dir = Path(processed_dir)
    csv_files = sorted(processed_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError("No csv files found in {}".format(processed_dir))

    feature_frame=[]
    #right now only builds match level features
    for file in csv_files:
        df = pd.read_csv(file)
        #datetime gets converted to string by pandas
        df["tourney_date"] = pd.to_datetime(df["tourney_date"], errors="coerce")

        df = add_serve_return_features(df)
        df = add_pre_match_context_features(df)
        df = add_post_match_context_features(df)


        feature_frame.append(df)

    #ignore index prevents index collision when concatenating years
    all_years_df = pd.concat(feature_frame, ignore_index=True)
    return all_years_df

#building features from df not csv file for white box test
def build_features_from_df(df: pd.DataFrame) -> pd.DataFrame:
    df = add_serve_return_features(df)
    df = add_pre_match_context_features(df)
    df = add_post_match_context_features(df)
    return df

if __name__ == "__main__":
    processed_dir = Path("data/processed/matches")

    print(f"Loading processed CSVs from: {processed_dir.resolve()}")

    df = build_features(processed_dir)

    print("\nsize:")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}\n")

    print("column names: ")
    for col in df.columns:
        print(col)

    print("\nDone!")


