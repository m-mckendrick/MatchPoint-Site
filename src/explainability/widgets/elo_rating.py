from src.explainability.narrative import narrative_elo

def build_widgets(row_model, row_full, shap_group):

    p1_name = row_full["p1_name"]
    p2_name = row_full["p2_name"]
    surface = row_full["surface_raw"]

    # Narrative text (SHAP-driven)
    narrative_text = narrative_elo(
        shap_group=shap_group,
        p1_name=p1_name,
        p2_name=p2_name,
        surface=surface
    )

    # Extract values
    p1_global = row_model["p1_elo_global"]
    p2_global = row_model["p2_elo_global"]
    diff_global = row_model["elo_global_diff"]

    p1_surface = row_model["p1_elo_surface"]
    p2_surface = row_model["p2_elo_surface"]
    diff_surface = row_model["elo_surface_diff"]

    p1_name = row_full["p1_name"]
    p2_name = row_full["p2_name"]
    surface = row_full["surface_raw"]

    return [

        {
            "type": "group_narrative",
            "title": "Elo Ratings Overview",
            "group": "Elo Ratings",
            "text": narrative_text
        },

        {
            "type": "comparison",
            "title": f"Elo Strength ({p1_name} vs {p2_name})",
            "icon": "🌍",
            "group": "Elo Ratings",
            "data": {
                "label_p1": p1_name,
                "label_p2": p2_name,
                "p1_Elo": p1_global,
                "p2_Elo": p2_global,
                "diff": diff_global
            }
        },

        {
            "type": "comparison",
            "title": f"{surface} Elo Strength ({p1_name} vs {p2_name})",
            "icon": "🎾",
            "group": "Elo Ratings",
            "data": {
                "label_p1": p1_name,
                "label_p2": p2_name,
                f"p1_{surface}_Elo": p1_surface,
                f"p2_{surface}_Elo": p2_surface,
                "diff": diff_surface
            }
        },

        {
            "type": "shap_bar",
            "title": "SHAP Impact – Elo",
            "icon": "📊",
            "group": "Elo Ratings",
            "data": shap_group
        }
    ]
