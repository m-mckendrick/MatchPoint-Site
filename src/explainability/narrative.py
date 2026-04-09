#very simple narrative for prototype
def build_narrative(top_features, winner):
    parts = []

    for f in top_features:
        name = f["feature"]

        if "rank" in name:
            parts.append("ranking difference was a major factor")
        elif "surface" in name:
            parts.append("surface suitability influenced the prediction")
        elif "age" in name:
            parts.append("age difference contributed to the outcome")
        elif "ht" in name:
            parts.append("player physique had some impact")

    if not parts:
        return f"The model predicted {winner} based on several contributing factors "

    return f"The model predicts {winner} mainly because " + ", ".join(parts) + "."
