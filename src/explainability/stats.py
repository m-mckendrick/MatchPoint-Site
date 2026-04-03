#extract all the diff stats
def extract_stats(row_df):
    row = row_df.iloc[0]
    stats = {}

    for col in row_df.columns:
        if "diff" in col:
            stats[col] = float(row[col])

    return stats
