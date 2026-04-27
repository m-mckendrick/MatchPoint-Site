from src.explainability.narrative import narrative_serve_return

def build_widgets(row_model, row_full, shap_group):

    p1_name = row_full["p1_name"]
    p2_name = row_full["p2_name"]

    # ============================
    # NARRATIVE
    # ============================
    narrative_text = narrative_serve_return(
        shap_group=shap_group,
        p1_name=p1_name,
        p2_name=p2_name
    )

    # ============================
    # ALL‑SURFACE SERVE FORM (5 MATCHES)
    # ============================

    p1_1st_srv_pct = row_model["p1_form_all_5_first_serve_pct"] * 100
    p2_1st_srv_pct = row_model["p2_form_all_5_first_serve_pct"] * 100

    p1_1st_srv_won = row_model["p1_form_all_5_first_serve_won_pct"] * 100
    p2_1st_srv_won = row_model["p2_form_all_5_first_serve_won_pct"] * 100

    p1_2nd_srv_won = row_model["p1_form_all_5_second_serve_won_pct"] * 100
    p2_2nd_srv_won = row_model["p2_form_all_5_second_serve_won_pct"] * 100

    p1_df_rate = row_model["p1_form_all_5_double_fault_rate"] * 100
    p2_df_rate = row_model["p2_form_all_5_double_fault_rate"] * 100

    p1_aces = row_model["p1_form_all_5_aces_per_game"]
    p2_aces = row_model["p2_form_all_5_aces_per_game"]

    p1_hold_pct = row_model["p1_form_all_5_estimated_hold_pct"] * 100
    p2_hold_pct = row_model["p2_form_all_5_estimated_hold_pct"] * 100

    p1_srv_eff = row_model["p1_form_all_5_serve_efficiency"] * 100
    p2_srv_eff = row_model["p2_form_all_5_serve_efficiency"] * 100


    # ============================
    # SURFACE‑SPECIFIC SERVE FORM (5 MATCHES)
    # ============================

    p1_1st_srv_pct_surf = row_model["p1_form_same_5_first_serve_pct"] * 100
    p2_1st_srv_pct_surf = row_model["p2_form_same_5_first_serve_pct"] * 100

    p1_1st_srv_won_surf = row_model["p1_form_same_5_first_serve_won_pct"] * 100
    p2_1st_srv_won_surf = row_model["p2_form_same_5_first_serve_won_pct"] * 100

    p1_2nd_srv_won_surf = row_model["p1_form_same_5_second_serve_won_pct"] * 100
    p2_2nd_srv_won_surf = row_model["p2_form_same_5_second_serve_won_pct"] * 100

    p1_df_rate_surf = row_model["p1_form_same_5_double_fault_rate"] * 100
    p2_df_rate_surf = row_model["p2_form_same_5_double_fault_rate"] * 100

    p1_aces_surf = row_model["p1_form_same_5_aces_per_game"]
    p2_aces_surf = row_model["p2_form_same_5_aces_per_game"]

    p1_hold_pct_surf = row_model["p1_form_same_5_estimated_hold_pct"] * 100
    p2_hold_pct_surf = row_model["p2_form_same_5_estimated_hold_pct"] * 100

    p1_srv_eff_surf = row_model["p1_form_same_5_serve_efficiency"] * 100
    p2_srv_eff_surf = row_model["p2_form_same_5_serve_efficiency"] * 100


    # ============================
    # ALL‑SURFACE RETURN FORM (5 MATCHES)
    # ============================

    p1_ret_pts_won = row_model["p1_form_all_5_return_pts_won_pct"] * 100
    p2_ret_pts_won = row_model["p2_form_all_5_return_pts_won_pct"] * 100

    p1_brk_pct = row_model["p1_form_all_5_estimated_break_pct"] * 100
    p2_brk_pct = row_model["p2_form_all_5_estimated_break_pct"] * 100


    # ============================
    # SURFACE‑SPECIFIC RETURN FORM (5 MATCHES)
    # ============================

    p1_ret_pts_won_surf = row_model["p1_form_same_5_return_pts_won_pct"] * 100
    p2_ret_pts_won_surf = row_model["p2_form_same_5_return_pts_won_pct"] * 100

    p1_brk_pct_surf = row_model["p1_form_same_5_estimated_break_pct"] * 100
    p2_brk_pct_surf = row_model["p2_form_same_5_estimated_break_pct"] * 100

    # ============================
    # WIDGETS
    # ============================

    return [

        # --- Narrative widget ---
        {
            "type": "group_narrative",
            "title": "Serve & Return Overview",
            "group": "Serve & Return",
            "text": narrative_text
        },

        # ============================
        # SERVE QUALITY – ALL SURFACES
        # ============================
        {
            "type": "comparison",
            "title": f"Serve Quality – All Surfaces ({p1_name} vs {p2_name})",
            "icon": "🎾",
            "group": "Serve & Return",
            "data": {
                "label_p1": p1_name,
                "label_p2": p2_name,

                "p1_1st_srv_pct": p1_1st_srv_pct,
                "p2_1st_srv_pct": p2_1st_srv_pct,

                "p1_1st_srv_won": p1_1st_srv_won,
                "p2_1st_srv_won": p2_1st_srv_won,

                "p1_2nd_srv_won": p1_2nd_srv_won,
                "p2_2nd_srv_won": p2_2nd_srv_won,

                "p1_df_rate": p1_df_rate,
                "p2_df_rate": p2_df_rate,

                "p1_aces": p1_aces,
                "p2_aces": p2_aces,

                "p1_hold_pct": p1_hold_pct,
                "p2_hold_pct": p2_hold_pct,

                "p1_srv_eff": p1_srv_eff,
                "p2_srv_eff": p2_srv_eff
            }
        },

        # ============================
        # SERVE QUALITY – SURFACE SPECIFIC
        # ============================
        {
            "type": "comparison",
            "title": f"Serve Quality – Surface Specific ({p1_name} vs {p2_name})",
            "icon": "🎾",
            "group": "Serve & Return",
            "data": {
                "label_p1": p1_name,
                "label_p2": p2_name,

                "p1_1st_srv_pct": p1_1st_srv_pct_surf,
                "p2_1st_srv_pct": p2_1st_srv_pct_surf,

                "p1_1st_srv_won": p1_1st_srv_won_surf,
                "p2_1st_srv_won": p2_1st_srv_won_surf,

                "p1_2nd_srv_won": p1_2nd_srv_won_surf,
                "p2_2nd_srv_won": p2_2nd_srv_won_surf,

                "p1_df_rate": p1_df_rate_surf,
                "p2_df_rate": p2_df_rate_surf,

                "p1_aces": p1_aces_surf,
                "p2_aces": p2_aces_surf,

                "p1_hold_pct": p1_hold_pct_surf,
                "p2_hold_pct": p2_hold_pct_surf,

                "p1_srv_eff": p1_srv_eff_surf,
                "p2_srv_eff": p2_srv_eff_surf
            }
        },

        # ============================
        # RETURN QUALITY – ALL SURFACES
        # ============================
        {
            "type": "comparison",
            "title": f"Return Quality – All Surfaces ({p1_name} vs {p2_name})",
            "icon": "↩️",
            "group": "Serve & Return",
            "data": {
                "label_p1": p1_name,
                "label_p2": p2_name,

                "p1_ret_pts_won": p1_ret_pts_won,
                "p2_ret_pts_won": p2_ret_pts_won,

                "p1_brk_pct": p1_brk_pct,
                "p2_brk_pct": p2_brk_pct
            }
        },

        # ============================
        # RETURN QUALITY – SURFACE SPECIFIC
        # ============================
        {
            "type": "comparison",
            "title": f"Return Quality – Surface Specific ({p1_name} vs {p2_name})",
            "icon": "↩️",
            "group": "Serve & Return",
            "data": {
                "label_p1": p1_name,
                "label_p2": p2_name,

                "p1_ret_pts_won": p1_ret_pts_won_surf,
                "p2_ret_pts_won": p2_ret_pts_won_surf,

                "p1_brk_pct": p1_brk_pct_surf,
                "p2_brk_pct": p2_brk_pct_surf
            }
        },

        # --- SHAP ---
        {
            "type": "shap_bar",
            "title": "SHAP Impact – Serve & Return",
            "icon": "📊",
            "group": "Serve & Return",
            "data": shap_group
        }
    ]