def build_player_profiles(row_df):
    row = row_df.iloc[0]

    p1 = {
        "name": row.get("p1_name", "Player 1"),
        "rank": row.get("p1_rank", None),
        "age": row.get("p1_age", None),
        "height": row.get("p1_ht", None),
        "hand": row.get("p1_hand", None)
    }

    p2 = {
        "name": row.get("p2_name", "Player 2"),
        "rank": row.get("p2_rank", None),
        "age": row.get("p2_age", None),
        "height": row.get("p2_ht", None),
        "hand": row.get("p2_hand", None)
    }

    return {"p1": p1, "p2": p2}
