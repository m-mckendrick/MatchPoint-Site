def build_player_profiles(row_df):
    row = row_df.iloc[0]

    def to_int(x):
        try:
            return int(x)
        except:
            return None



    p1 = {
        "name": row.get("p1_name", "Player 1"),
        "rank": to_int(row.get("p1_rank", None)),
        "age": to_int(row.get("p1_age", None)),
        "height": to_int(row.get("p1_ht", None)),
        "hand": row .get("p1_hand", None),
        "ioc": row.get("p1_ioc",None),

        # Elo
        "elo_global": to_int(row.get("p1_elo_global",None)),
        "elo_surface": to_int(row.get("p1_elo_surface",None)),

    }

    p2 = {
        "name": row.get("p2_name", "Player 2"),
        "rank": to_int(row.get("p2_rank", None)),
        "age": to_int(row.get("p2_age", None)),
        "height": to_int(row.get("p2_ht", None)),
        "hand": row.get("p2_hand", None),
        "ioc": row.get("p1_ioc",None),

        # Elo
        "elo_global": to_int(row.get("p1_elo_global",None)),
        "elo_surface": to_int(row.get("p1_elo_surface",None)),
    }

    return {"p1": p1, "p2": p2}
