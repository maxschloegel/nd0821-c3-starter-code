# Put the code for your API here.
import json
import pickle

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
import pandas as pd
from pydantic import BaseModel, Field

from starter.ml.config import cat_features
from starter.ml.model import inference_on_df


app = FastAPI()


class InferenceRequest(BaseModel):
    age: int = Field(example=39)
    workclass: str = Field(example=" State-gov")
    fnlgt: int = Field(example=77516)
    education: str = Field(example="Bachelors")
    education_num: int = Field(example=13, alias="education-num")
    marital_status: str = Field(example="Never-married", alias="marital-status")
    occupation: str = Field(exapmle="Adm-clerical")
    relationship: str = Field(example="Not-in-family")
    race: str = Field(example="White")
    sex: str = Field(example="Male")
    capital_gain: int = Field(example=2174, alias="capital-gain")
    capital_loss: int = Field(example=0, alias="capital-loss")
    hours_per_week: int = Field(example=40, alias="hours-per-week")
    native_country: str = Field(example="United-States", alias="native-country")
    salary: str = Field(example="<=50K")

    

#    model_config = {
#        "json_schema_extra": {
#            "examples": [
#                {
#                    "model_name": "classifier.pkl",
#                    "df_json": ("""{"age":{"0":39},\
#                                 "workclass":{"0":" State-gov"},\
#                                 "fnlgt":{"0":77516},\
#                                 "education":{"0":" Bachelors"},\
#                                 "education-num":{"0":13},\
#                                 "marital-status":{"0":" Never-married"},\
#                                 "occupation":{"0":" Adm-clerical"},\
#                                 "relationship":{"0":" Not-in-family"},\
#                                 "race":{"0":" White"},\
#                                 "sex":{"0":" Male"},\
#                                 "capital-gain":{"0":2174},\
#                                 "capital-loss":{"0":0},\
#                                 "hours-per-week":{"0":40},\
#                                 "native-country":{"0":" United-States"},\
#                                 "salary":{"0":" <=50K"}}'""")
#                    }
#                ]
#            }
#        }



@app.get("/")
async def say_hello():
    return {"greeting": "Welcome to my API!"}


@app.post("/inference")
async def inference_df(request: InferenceRequest):
    # load model
    with open(f"model/classifier.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/one_hot_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    with open("model/lb.pkl", "rb") as f:
        lb = pickle.load(f)

#    df = pd.DataFrame.read_json(request.df_json)
    df = pd.DataFrame(jsonable_encoder(request), index=[0])
    print(df)
    predictions = inference_on_df(model=model,
                                  df=df,
                                  categorical_features=cat_features,
                                  encoder=encoder,
                                  lb=lb)

    # Turn binary label into original category-classes
    predictions = lb.inverse_transform(predictions)
    # Cast predictions into json-format
    predictions = json.dumps(list(predictions))

    return {"predictions": predictions}
