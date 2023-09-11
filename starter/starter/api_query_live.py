import json
import requests


data = {
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

data_json = json.dumps(data)
#curl -X 'POST' \
#  'https://income-prediction-hwg8.onrender.com/inference' \
#  -H 'accept: application/json' \
#  -H 'Content-Type: application/json' \
#  -d data_json

url = 'https://income-prediction-hwg8.onrender.com/inference'
headers = {'content-type': 'application/json', 'accept': 'application/json'}

response = requests.post(url, data=data_json, headers=headers)

print(f"Status Code: {response.status_code}")
print(f"Prediction Result: {response.json()}")
