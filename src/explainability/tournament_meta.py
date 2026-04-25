import pandas as pd

def extract_tournament_meta(row_df):
    row = row_df.iloc[0]

    #added so that in main.py (api) nupy type arnt used instead of int
    def to_int(x):
        return int(x) if x is not None and not pd.isna(x) else None

    def to_float(x):
        return float(x) if x is not None and not pd.isna(x) else None

    return {
        "tournament": row.get("tourney_name", None),
        "year": pd.to_datetime(row.get("tourney_date")).year if row.get("tourney_date") else None,
        "round": row.get("round", None),
        "surface": row.get("surface_raw", None),

        "country": row.get("tourney_country", None),

        "draw_size": to_int(row.get("draw_size", None)),
        "best_of": to_int(row.get("best_of", None)),
        "is_indoor": bool(row.get("is_indoor", False)) if not pd.isna(row.get("is_indoor", None)) else None,

        "is_grand_slam": bool(row.get("is_grand_slam", False)),
        "is_masters": bool(row.get("is_masters", False)),
        "stage_pressure": to_float(row.get("stage_pressure", None)),
    }
