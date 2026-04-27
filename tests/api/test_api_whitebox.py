def test_api_predict_numpy_conversion(client):
    payload = {
        "p1_name": "Novak Djokovic",
        "p2_name": "Rafael Nadal",
        "tournament": "Wimbledon",
        "year": 2023,
        "round_label": "QF"
    }

    data = client.post("/api/predict", json=payload).json()

    def check_no_numpy(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                check_no_numpy(v)
        elif isinstance(obj, list):
            for v in obj:
                check_no_numpy(v)
        else:
            assert not str(type(obj)).startswith("<class 'numpy.")

    check_no_numpy(data)


def test_api_predict_winner_logic(monkeypatch, client):
    # Force model to return a fixed probability
    monkeypatch.setattr("src.prediction.predictor.predict", lambda m, s, r: 0.9)

    payload = {
        "p1_name": "Novak Djokovic",
        "p2_name": "Rafael Nadal",
        "tournament": "Wimbledon",
        "year": 2023,
        "round_label": "R32"
    }

    data = client.post("/api/predict", json=payload).json()
    assert data["prediction"]["winner"] == "Novak Djokovic"


def test_api_invalid_year(client):
    """Invalid year should be rejected by validation."""
    payload = {
        "p1_name": "N.Djokovic",
        "p2_name": "C.Alcaraz",
        "tournament": "Wimbledon",
        "year": 1800,
        "round_label": "R32"
    }

    response = client.post("/api/predict", json=payload)
    assert response.status_code == 400


def test_api_invalid_round(client):
    """Invalid round label should be rejected."""
    payload = {
        "p1_name": "N.Djokovic",
        "p2_name": "C.Alcaraz",
        "tournament": "Wimbledon",
        "year": 2023,
        "round_label": "NotARound"
    }

    response = client.post("/api/predict", json=payload)
    assert response.status_code == 400


def test_api_handles_empty_payload(client):
    """Empty payload should return 422 (Pydantic validation)."""
    response = client.post("/api/predict", json={})
    assert response.status_code in (400, 422)
