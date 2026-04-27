import pandas as pd
import numpy as np

from data.loader import PROCESSED_PATH, MATCHES_PATH
from src.model_training.dataset_preparation import prepare_dataset


def inspect_dataset(df):

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.max_seq_items", None)


    print("\n=== LOADING CSV ===")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {df.shape[1]}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")

    #missing values
    print("\n=== MISSING VALUES (%) ===")
    missing_pct = df.isna().mean().sort_values(ascending=False) * 100
    print(missing_pct[missing_pct > 0])

    #percentage of 0 values
    print("\n=== ZERO VALUES (%) ===")
    zero_pct = (df == 0).mean().sort_values(ascending=False) * 100
    print(zero_pct[zero_pct > 20])

    #constant columns
    print("\n=== CONSTANT COLUMNS (all values identical) ===")
    constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) == 1]
    print(constant_cols)

    #near constant
    print("\n=== NEAR-CONSTANT COLUMNS (>99% same value) ===")
    near_constant = []
    for col in df.columns:
        top_freq = df[col].value_counts(normalize=True, dropna=False).max()
        if top_freq > 0.99:
            near_constant.append((col, round(top_freq * 100, 2)))
    print(near_constant)

    #duplicate rows
    print("\n=== DUPLICATE ROWS ===")
    print(df.duplicated().sum())

    #datatype summary
    print("\n=== DATA TYPES ===")
    print(df.dtypes.value_counts())

    #basic stats summary
    print("\n=== NUMERIC SUMMARY (first 10 columns) ===")
    print(df.describe().T.head(10))

    #suspicious columns
    print("\n=== SUSPICIOUS COLUMNS (all zero or all NaN) ===")
    suspicious = []
    for col in df.columns:
        if df[col].isna().all() or (df[col] == 0).all():
            suspicious.append(col)
    print(suspicious)


    print("\n=== DONE ===")


if __name__ == "__main__":
    print("works")
    output_path = PROCESSED_PATH / f"2000-2023_features.csv"
    df = pd.read_csv(output_path)
    #df = pd.read_csv(MATCHES_PATH)
    df = prepare_dataset(df)
    inspect_dataset(df)


