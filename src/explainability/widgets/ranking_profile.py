from src.explainability.narrative import narrative_profile_ranking

def build_widgets(row_model, row_full, shap_group):

    p1_name = row_full["p1_name"]
    p2_name = row_full["p2_name"]

    # Narrative text (SHAP-driven)
    narrative_text = narrative_profile_ranking(
        shap_group=shap_group,
        p1_name=p1_name,
        p2_name=p2_name
    )

    # Ranking values
    p1_rank = row_model["p1_rank"]
    p2_rank = row_model["p2_rank"]
    rank_diff = row_model["rank_diff"]

    p1_rank_points = row_model["p1_rank_points"]
    p2_rank_points = row_model["p2_rank_points"]
    rank_points_diff = row_model["rank_points_diff"]

    # Physical values
    p1_age = row_model["p1_age"]
    p2_age = row_model["p2_age"]
    p1_ht = row_model["p1_ht"]
    p2_ht = row_model["p2_ht"]
    p1_hand = "Left" if row_model["p1_lefty"] == 1 else "Right"
    p2_hand = "Left" if row_model["p2_lefty"] == 1 else "Right"

    return [

        {
            "type": "group_narrative",
            "title": "Player Profile & Ranking Overview",
            "group": "Player Profile & Ranking",
            "text": narrative_text
        },

        {
            "type": "comparison",
            "title": f"Ranking Comparison ({p1_name} vs {p2_name})",
            "icon": "🏆",
            "group": "Player Profile & Ranking",
            "data": {
                "label_p1": p1_name,
                "label_p2": p2_name,
                "p1_rank": p1_rank,
                "p2_rank": p2_rank,
                "rank_diff": rank_diff,
                "p1_rank_points": p1_rank_points,
                "p2_rank_points": p2_rank_points,
                "rank_points_diff": rank_points_diff
            }
        },

        {
            "type": "stat_card",
            "title": f"Physical Profile ({p1_name} vs {p2_name})",
            "icon": "📏",
            "group": "Player Profile & Ranking",
            "data": {
                "p1_age": p1_age,
                "p2_age": p2_age,
                "p1_height": p1_ht,
                "p2_height": p2_ht,
                "p1_hand": p1_hand,
                "p2_hand": p2_hand
            }
        },

        {
            "type": "shap_bar",
            "title": "SHAP Impact – Profile & Ranking",
            "icon": "📊",
            "group": "Player Profile & Ranking",
            "data": shap_group
        }
    ]
