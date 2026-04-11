import pandas as pd

def extract_tournament_meta(row_df):
    row = row_df.iloc[0]

    #added so that in main.py (api) nupy type arnt used instead of int
    def to_int(x):
        return int(x) if x is not None and not pd.isna(x) else None

    return {
        "tournament": row.get("tourney_name", None),
        "year": pd.to_datetime(row.get("tourney_date")).year if row.get("tourney_date") else None,
        "round": row.get("round", None),
        "surface": row.get("surface_raw", None)
    }
