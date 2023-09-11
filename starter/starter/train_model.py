# Script to train machine learning model.
import pathlib

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import clean_data, load_data_path, process_data
from ml.model import (save_encoder, save_model, save_lb, train_model,
                      compute_model_metrics, inference,
                      compute_model_metrics_for_feature_slice)
from ml.config import cat_features, label

# Add the necessary imports for the starter code.

# Add code to load in the data.
data_path = load_data_path()
print(data_path)
data = pd.read_csv(data_path)

# Clean data (remove nan values or weird entries such as "?")
data = clean_data(data)

# Optional: use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

# Proces the test data with the process_data function.
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label=label, training=True
)
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features,
    label=label, training=False, encoder=encoder, lb=lb
)

# Train and save a model.
model = train_model(X_train, y_train)

# Computing Metrics
train_metrics = compute_model_metrics(inference(model, X_train), y_train)
test_metrics = compute_model_metrics(inference(model, X_test), y_test)
slice_metrics_sex = compute_model_metrics_for_feature_slice(
                        model=model,
                        df=test,
                        categorical_features=cat_features,
                        label=label,
                        slice_column='sex',
                        encoder=encoder,
                        lb=lb)
slice_metrics_education = compute_model_metrics_for_feature_slice(
                        model=model,
                        df=test,
                        categorical_features=cat_features,
                        label=label,
                        slice_column='education',
                        encoder=encoder,
                        lb=lb)


def create_metrics_table_row(metrics, name):
    string = (f"|{name: <30}|{metrics[0]:11.3f}|"
              f"{metrics[1]:8.3f}|{metrics[2]:10.3f}|\n")
    return string


def create_metrics_row_for_slice(metrics, name):
    slice_metric_string = ""
    for k, v in metrics.items():
        slice_metric_string += create_metrics_table_row(v, name+f"-{k}")

    return slice_metric_string


with open("slice_output.txt", "w") as f:
    f.write(f"|{'Data': <30}| Precision | Recall | F1-Score |\n")
    f.write("|"+30*'-'+"|-----------|--------|----------|\n")
    f.write(create_metrics_table_row(train_metrics, "Train"))
    f.write(create_metrics_table_row(test_metrics, "Test"))
    f.write(create_metrics_row_for_slice(slice_metrics_sex, "Slice-Sex"))
    f.write(create_metrics_row_for_slice(slice_metrics_education,
                                         "Slice-Education"))


# Save model
model_path = pathlib.Path("model") / "classifier.pkl"
save_model(model, model_path)

# Save encoder
encoder_path = pathlib.Path("model") / "one_hot_encoder.pkl"
save_encoder(encoder, encoder_path)

# Save labelbinarizer
lb_path = pathlib.Path("model") / "lb.pkl"
save_lb(lb, lb_path)
