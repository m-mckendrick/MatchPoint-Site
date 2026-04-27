def test_api_predict_valid(client):
    payload = {
        "p1_name": "Novak Djokovic",
        "p2_name": "Rafael Nadal",
        "tournament": "Wimbledon",
        "year": 2023,
        "round_label": "QF"
    }

    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "prediction" in data
    assert "explainability" in data

def test_api_predict_structure(client):
    payload = {
        "p1_name": "Novak Djokovic",
        "p2_name": "Rafael Nadal",
        "tournament": "Wimbledon",
        "year": 2023,
        "round_label": "QF"
    }

    data = client.post("/api/predict", json=payload).json()

    assert "winner" in data["prediction"]
    assert "probability" in data["prediction"]

    explain = data["explainability"]
    assert "players" in explain
    assert "tournament" in explain
    assert "narrative" in explain
    assert "widgets" in explain


def test_api_invalid_player_returns_400(client):
    """Invalid players should trigger validation and return 400."""
    payload = {
        "p1_name": "FakePlayer",
        "p2_name": "AnotherFake",
        "tournament": "Wimbledon",
        "year": 2023,
        "round_label": "R32"
    }

    response = client.post("/api/predict", json=payload)
    assert response.status_code == 400
    assert "error" in response.json()


def test_api_invalid_tournament_returns_400(client):
    """Invalid tournament should return 400."""
    payload = {
        "p1_name": "N.Djokovic",
        "p2_name": "C.Alcaraz",
        "tournament": "FakeTournament",
        "year": 2023,
        "round_label": "R32"
    }

    response = client.post("/api/predict", json=payload)
    assert response.status_code == 400


def test_home_page_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_results_page_served(client):
    response = client.get("/results")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]