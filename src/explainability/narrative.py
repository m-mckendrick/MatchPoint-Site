#very simple narrative for prototype
#this will be changed later to take the stats types into account and create a narrative based on what feature is the most popular

def narrative_profile_ranking(shap_group, p1_name, p2_name):
    top_feature = max(shap_group["raw_shap_values"], key=lambda k: abs(shap_group["raw_shap_values"][k]))

    return (
        f"Ranking and physical attributes played a role in this matchup. "
        f"The most influential factor was {top_feature.replace('_', ' ')}, "
        f"highlighting how long‑term ability and physical traits compare between {p1_name} and {p2_name}."
    )


def narrative_elo(shap_group, p1_name, p2_name, surface):
    top_feature = max(shap_group["raw_shap_values"], key=lambda k: abs(shap_group["raw_shap_values"][k]))

    return (
        f"Elo Ratings were an important factor. The strongest influence came from {top_feature.replace('_', ' ')}, "
        f"showing how {p1_name} and {p2_name} differ in both global strength and performance on {surface}."
    )



def narrative_serve_return(shap_group, p1_name, p2_name):
    top_feature = max(shap_group["raw_shap_values"], key=lambda k: abs(shap_group["raw_shap_values"][k]))

    return (
        f"Serve and return performance had a meaningful impact. "
        f"The most influential stat was {top_feature.replace('_', ' ')}, "
        f"highlighting differences in how {p1_name} and {p2_name} hold serve and pressure opponents on return."
    )



def narrative_pressure(shap_group, p1_name, p2_name):
    top_feature = max(shap_group["raw_shap_values"], key=lambda k: abs(shap_group["raw_shap_values"][k]))

    return (
        f"Clutch performance under pressure influenced the prediction. "
        f"The key factor was {top_feature.replace('_', ' ')}, "
        f"reflecting how {p1_name} and {p2_name} perform in high‑leverage moments such as break points."
    )


def narrative_fatigue(shap_group, p1_name, p2_name):
    top_feature = max(shap_group["raw_shap_values"], key=lambda k: abs(shap_group["raw_shap_values"][k]))

    return (
        f"Fatigue contributed to the prediction. "
        f"The most impactful fatigue metric was {top_feature.replace('_', ' ')}, "
        f"indicating how physical load differed between {p1_name} and {p2_name} entering the match."
    )

def narrative_context(shap_group, p1_name, p2_name, surface, round_label):
    top_feature = max(shap_group["raw_shap_values"], key=lambda k: abs(shap_group["raw_shap_values"][k]))

    return (
        f"Match context played a role in shaping expectations. "
        f"The most influential contextual factor was {top_feature.replace('_', ' ')}, "
        f"reflecting how surface ({surface}), round ({round_label}), and tournament conditions "
        f"affect the matchup between {p1_name} and {p2_name}."
    )



def build_narrative_groups(grouped_shap, winner):
    if not grouped_shap:
        return f"The model predicts {winner} based on several contributing factors."

    groups_sorted = sorted(
        grouped_shap.items(),
        key=lambda x: x[1]["sum_abs_shap"],
        reverse=True
    )
    # The top feature is the first one in the list
    top_groups = [g[0] for g in groups_sorted[:3]]

    # Build readable text
    if len(top_groups) == 1:
        return (
            f"The model predicts {winner} mainly because the "
            f"{top_groups[0].lower()} features had the strongest influence on the prediction."
        )

    if len(top_groups) == 2:
        return (
            f"The model predicts {winner} primarily due to the influence of "
            f"{top_groups[0].lower()} and {top_groups[1].lower()}."
        )

    # 3 groups
    return (
        f"The model predicts {winner} based on a combination of "
        f"{top_groups[0].lower()}, {top_groups[1].lower()}, and {top_groups[2].lower()}, "
        f"which were the most impactful feature groups for this matchup."
    )


