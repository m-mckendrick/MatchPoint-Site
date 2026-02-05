import pandas as pd
from pathlib import Path

from .serve_return import add_serve_return_features
from .match_context import add_match_context_features

def build_features(processed_dir="data/processed"):
    #get all csv that have been cleaned
    #coverts folder path to folder object allows year sorting etc.
    processed_dir = Path(processed_dir)
    csv_files = sorted(processed_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError("No csv files found in {}".format(processed_dir))

    feature_frame=[]

    for file in csv_files:
        df = pd.read_csv(file)
        df = add_serve_return_features(df)
        df = add_match_context_features(df)

        #creates target variable for model
        #atm im using a winner perspective dataset so label is always 1 (1==winner)
        #later we may implement loser rows too
        df["label"] = 1

        feature_frame.append(df)

    all_years_df = pd.concat(feature_frame)
    return all_years_df

#building features from df not csv file for white box test
def build_features_from_df(df: pd.DataFrame) -> pd.DataFrame:
    df = add_serve_return_features(df)
    df = add_match_context_features(df)
    df["label"] = 1
    return df

