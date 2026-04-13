import pandas as pd
from pathlib import Path

from .serve_return import add_serve_return_features
from .match_context import add_pre_match_context_features
from .match_context import add_post_match_context_features

# df acts as both the synthetic row and all the processed matches
#full df is always the full set of processed matches
def build_features(df: pd.DataFrame, full_df: pd.DataFrame = None) -> pd.DataFrame:

    df = add_pre_match_context_features(df)
    df = add_post_match_context_features(df)
    # df = add_serve_return_features(df)
    #full df allows in prediction the whole dataset to be pass through so that feature that require looking at previous matches can happend

    return df


if __name__ == "__main__":

    #df = build_features(load_processed_matches)

    print("\nsize:")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}\n")

    print("column names: ")
    for col in df.columns:
        print(col)

    print("\nDone!")


