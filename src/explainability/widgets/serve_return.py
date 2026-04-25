from src.explainability.narrative import narrative_serve_return

def build_widgets(row_model, row_full, shap_group):

    p1_name = row_full["p1_name"]
    p2_name = row_full["p2_name"]


    narrative_text = narrative_serve_return(
        shap_group=shap_group,
        p1_name=p1_name,
        p2_name=p2_name
    )

    #stats
    p1_first_serve_pct = row_model["p1_form_all_5_first_serve_pct"]
    p2_first_serve_pct = row_model["p2_form_all_5_first_serve_pct"]

    p1_first_serve_won = row_model["p1_form_all_5_first_serve_won_pct"]
    p2_first_serve_won = row_model["p2_form_all_5_first_serve_won_pct"]

    p1_second_serve_won = row_model["p1_form_all_5_second_serve_won_pct"]
    p2_second_serve_won = row_model["p2_form_all_5_second_serve_won_pct"]

    p1_df_rate = row_model["p1_form_all_5_double_fault_rate"]
    p2_df_rate = row_model["p2_form_all_5_double_fault_rate"]

    p1_aces = row_model["p1_form_all_5_aces_per_game"]
    p2_aces = row_model["p2_form_all_5_aces_per_game"]

    p1_hold = row_model["p1_form_all_5_estimated_hold_pct"]
    p2_hold = row_model["p2_form_all_5_estimated_hold_pct"]

    p1_serve_eff = row_model["p1_form_all_5_serve_efficiency"]
    p2_serve_eff = row_model["p2_form_all_5_serve_efficiency"]


    p1_return_pts = row_model["p1_form_all_5_return_pts_won_pct"]
    p2_return_pts = row_model["p2_form_all_5_return_pts_won_pct"]

    p1_break_pct = row_model["p1_form_all_5_estimated_break_pct"]
    p2_break_pct = row_model["p2_form_all_5_estimated_break_pct"]

    return [

        # --- Narrative widget ---
        {
            "type": "group_narrative",
            "title": "Serve & Return Overview",
            "group": "Serve & Return",
            "text": narrative_text
        },

        {
            "type": "comparison",
            "title": f"Serve Quality ({p1_name} vs {p2_name})",
            "icon": "🎾",
            "group": "Serve & Return",
            "data": {
                "label_p1": p1_name,
                "label_p2": p2_name,

                "p1_first_serve_pct": p1_first_serve_pct,
                "p2_first_serve_pct": p2_first_serve_pct,

                "p1_first_serve_won": p1_first_serve_won,
                "p2_first_serve_won": p2_first_serve_won,

                "p1_second_serve_won": p1_second_serve_won,
                "p2_second_serve_won": p2_second_serve_won,

                "p1_double_fault_rate": p1_df_rate,
                "p2_double_fault_rate": p2_df_rate,

                "p1_aces_per_game": p1_aces,
                "p2_aces_per_game": p2_aces,

                "p1_hold_pct": p1_hold,
                "p2_hold_pct": p2_hold,

                "p1_serve_efficiency": p1_serve_eff,
                "p2_serve_efficiency": p2_serve_eff
            }
        },

        {
            "type": "comparison",
            "title": f"Return Quality ({p1_name} vs {p2_name})",
            "icon": "↩️",
            "group": "Serve & Return",
            "data": {
                "label_p1": p1_name,
                "label_p2": p2_name,

                "p1_return_pts_won": p1_return_pts,
                "p2_return_pts_won": p2_return_pts,

                "p1_break_pct": p1_break_pct,
                "p2_break_pct": p2_break_pct
            }
        },

        {
            "type": "shap_bar",
            "title": "SHAP Impact – Serve & Return",
            "icon": "📊",
            "group": "Serve & Return",
            "data": shap_group
        }
    ]
