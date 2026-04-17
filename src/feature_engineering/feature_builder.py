import pandas as pd
from pathlib import Path

from .match_context import add_pre_match_context_features
from .match_results import add_match_result_features
from .serve_return import add_serve_return_features
from .pressure_performance import add_pressure_performance_features

from data.loader import load_matches

# df acts as both the synthetic row and all the processed matches
#full df is always the full set of processed matches
def build_features(df: pd.DataFrame) -> pd.DataFrame:

    df = add_pre_match_context_features(df)
    df = add_serve_return_features(df)
    df = add_match_result_features(df)


    return df




if __name__ == "__main__":


    df=load_matches()

    df = build_features(df)

    print("\nsize:")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}\n")

    print("column names: ")
    for col in df.columns:
        print(col)

    print("\nDone!")


