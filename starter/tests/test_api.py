import pathlib

from fastapi.testclient import TestClient
import pandas as pd

from main import app

client = TestClient(app)

data_path = pathlib.Path("tests/data/test_data.csv")
df = pd.read_csv(data_path)


def test_api_root():
    response = client.get("/")

    print(response)
    assert response.status_code == 200
    assert response.json() == {"greeting": "Welcome to my API!"}


def test_model_inference_cl0():
    data_json = {
        "age": 39,
        "workclass": " State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "string",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
        "salary": "<=50K"
        }
    response = client.post("/inference", json=data_json)

    assert response.status_code == 200
    assert response.json()["predictions"] == '["<=50K"]'


def test_model_inference_cl1():
    data_json = {
        "age": 42,
        "workclass": "Private",
        "fnlgt": 159449,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 5178,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
        "salary": ">50K"
        }
    response = client.post("/inference", json=data_json)

    assert response.status_code == 200
    assert response.json()["predictions"] == '[">50K"]'
