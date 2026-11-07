from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "Fraud Detection API is running"


def test_fraud_prediction():
    fraud_transaction = {
        "Time": 406,
        "V1": -2.312226542,
        "V2": 1.951992011,
        "V3": -1.609850732,
        "V4": 3.997905588,
        "V5": -0.522187865,
        "V6": -1.426545319,
        "V7": -2.537387306,
        "V8": 1.391657248,
        "V9": -2.770089277,
        "V10": -2.772272145,
        "V11": 3.202033207,
        "V12": -2.899907388,
        "V13": -0.595221881,
        "V14": -4.289253782,
        "V15": 0.38972412,
        "V16": -1.14074718,
        "V17": -2.830055675,
        "V18": -0.016822468,
        "V19": 0.416955705,
        "V20": 0.126910559,
        "V21": 0.517232371,
        "V22": -0.035049369,
        "V23": -0.465211076,
        "V24": 0.320198199,
        "V25": 0.044519167,
        "V26": 0.177839798,
        "V27": 0.261145003,
        "V28": -0.143275875,
        "Amount": 0
    }

    response = client.post("/predict", json=fraud_transaction)

    assert response.status_code == 200
    assert response.json()["prediction"] == 1
    assert response.json()["result"] == "Fraud"
    assert response.json()["risk_level"] == "High"


def test_legitimate_prediction():
    legitimate_transaction = {
        # Replace this dictionary with the legitimate
        # transaction JSON generated in your notebook.
        
        "Time": 0.0,
        "V1": 1.19185711131486,
        "V2": 0.26615071205963,
        "V3": 0.16648011335321,
        "V4": 0.448154078460911,
        "V5": 0.0600176492822243,
        "V6": -0.0823608088155687,
        "V7": -0.0788029833323113,
        "V8": 0.0851016549148104,
        "V9": -0.255425128109186,
        "V10": -0.166974414004614,
        "V11": 1.61272666105479,
        "V12": 1.06523531137287,
        "V13": 0.48909501589608,
        "V14": -0.143772296441519,
        "V15": 0.635558093258208,
        "V16": 0.463917041022171,
        "V17": -0.114804663102346,
        "V18": -0.183361270123994,
        "V19": -0.145783041325259,
        "V20": -0.0690831352230203,
        "V21": -0.225775248033138,
        "V22": -0.638671952771851,
        "V23": 0.101288021253234,
        "V24": -0.339846475529127,
        "V25": 0.167170404418143,
        "V26": 0.125894532368176,
        "V27": -0.00898309914322813,
        "V28": 0.0147241691924927,
        "Amount": 2.69
    }
    

    response = client.post("/predict", json=legitimate_transaction)

    assert response.status_code == 200
    assert response.json()["prediction"] == 0
    assert response.json()["result"] == "Legitimate"
    assert response.json()["risk_level"] == "Low"


def test_missing_feature_returns_422():
    response = client.post("/predict", json={})

    assert response.status_code == 422


def test_negative_amount_returns_422():
    invalid_transaction = {
        "Time": -1,
        "Amount": -100
    }

    response = client.post("/predict", json=invalid_transaction)

    assert response.status_code == 422