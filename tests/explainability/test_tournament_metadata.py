from src.explainability.tournament_meta import extract_tournament_meta

def test_tournament_meta_basic(synthetic_row):
    meta = extract_tournament_meta(synthetic_row)

    assert isinstance(meta, dict)

    #check essential fields
    for key in ["tournament", "surface", "round", "year"]:
        assert key in meta
