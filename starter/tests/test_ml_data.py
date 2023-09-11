import pathlib

import numpy as np
import pandas as pd

from starter.ml import data
from starter.ml.data import process_data
from starter.ml.model import load_pkl, inference_on_df, compute_model_metrics_for_feature_slice
from starter.ml.config import cat_features, label

data_path = pathlib.Path("tests/data/test_data.csv")
df = pd.read_csv(data_path)

model_path = pathlib.Path("model/classifier.pkl")
encoder_path = pathlib.Path("model/one_hot_encoder.pkl")
lb_path = pathlib.Path("model/lb.pkl")

model = load_pkl(model_path)
encoder = load_pkl(encoder_path)
lb = load_pkl(lb_path)


def test_process_data_output_shape():
    X, y, _, _ = process_data(X=df,
                              categorical_features=cat_features,
                              label=label,
                              training=False,
                              encoder=encoder,
                              lb=lb)

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert len(X) == len(df)
    assert len(y) == len(df)


def test_inference_on_df():
    predictions = inference_on_df(model=model,
                                  df=df,
                                  categorical_features=cat_features,
                                  encoder=encoder,
                                  lb=lb)

    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(df)


def test_metrics_for_feature_slices():
    metric_dict = compute_model_metrics_for_feature_slice(
                    model=model,
                    df=df,
                    categorical_features=cat_features,
                    label=label,
                    slice_column="sex",
                    encoder=encoder,
                    lb=lb)

    assert set(metric_dict.keys()) == set(["Male", "Female"])
    assert len(metric_dict["Male"]) == 3 
