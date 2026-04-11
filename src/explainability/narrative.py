#very simple narrative for prototype
#this will be changed later to take the stats types into account and create a narrative based on what feature is the most popular
def build_narrative(top_features, winner):
    if not top_features:
        return f"The model predicts {winner} based on several contributing factors."

    # The top feature is the first one in the list
    top_feature = top_features[0]["feature"]

    # Make the feature name more readable
    readable = top_feature.replace("_", " ").strip()

    return f"The model predicts {winner} mainly because {readable} had the strongest impact on the prediction."