def extract_tournament_meta(row_df):
    row = row_df.iloc[0]

    return {
        "tournament": row.get("tourney_name", None),
        "surface": row.get("surface_raw", None),
        "round": row.get("round", None),
        "year": row.get("tourney_year", None)
    }
