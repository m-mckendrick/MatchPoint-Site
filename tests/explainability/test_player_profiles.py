from src.explainability.player_profiles import build_player_profiles

def test_player_profiles_basic(synthetic_row):
    profiles = build_player_profiles(synthetic_row)

    assert "p1" in profiles
    assert "p2" in profiles

    assert isinstance(profiles["p1"], dict)
    assert isinstance(profiles["p2"], dict)

    # Check essential fields exist
    for key in ["rank", "age", "height", "hand"]:
        assert key in profiles["p1"]
        assert key in profiles["p2"]
