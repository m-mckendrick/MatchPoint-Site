import pandas as pd
from src.feature_engineering.match_context import match_context

def test_match_context_features():
    df = pd.read_csv("../../data/processed/atp_matches_2024_clean.csv")
    df2 = match_context(df)

    #show all columns
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 300)

    #all new columns start after this
    start_index = 48
    new_cols = df2.columns[start_index:]

    print("\n COLUMNS ADDED (first 20 rows):\n")
    print(df2[new_cols].head(20).to_string())


if __name__ == "__main__":
    test_match_context_features()