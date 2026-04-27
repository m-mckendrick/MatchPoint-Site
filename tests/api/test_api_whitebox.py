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
        "p1_name": "PlayerA",
        "p2_name": "PlayerB",
        "tournament": "Wimbledon",
        "year": 2023,
        "round_label": "R32"
    }

    data = client.post("/api/predict", json=payload).json()
    assert data["prediction"]["winner"] == "PlayerA"

def test_api_predict_explainability_called(monkeypatch, client):
    called = {"flag": False}

    def fake_explain(*args, **kwargs):
        called["flag"] = True
        return {
            "players": {},
            "tournament": {},
            "narrative": "test",
            "widgets": []
        }

    monkeypatch.setattr("src.explainability.explainer.explain_prediction", fake_explain)

    payload = {
        "p1_name": "PlayerA",
        "p2_name": "PlayerB",
        "tournament": "Wimbledon",
        "year": 2023,
        "round_label": "R32"
    }

    client.post("/api/predict", json=payload)
    assert called["flag"] is True
