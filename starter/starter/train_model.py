# Script to train machine learning model.
import os
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

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

# Proces the test data with the process_data function.
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label=label, training=True
)
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label=label, training=False, encoder=encoder, lb=lb
)

# Train and save a model.
model = train_model(X_train, y_train)

print(f"Train Metrics - ({len(X_train)} samples used)")
print(compute_model_metrics(inference(model, X_train), y_train))
print(f"Test Metrics - ({len(X_test)} samples used)")
print(compute_model_metrics(inference(model, X_test), y_test))

print("Slice Metrics on Test data for Column 'sex'")
print(compute_model_metrics_for_feature_slice(model=model,
                                              df=test,
                                              categorical_features=cat_features,
                                              label=label,
                                              slice_column='sex',
                                              encoder=encoder,
                                              lb=lb))

#import pdb;pdb.set_trace()
# Save model
model_path = pathlib.Path("model") / "classifier.pkl"
save_model(model, model_path)

# Save encoder
encoder_path = pathlib.Path("model") / "one_hot_encoder.pkl"
save_encoder(encoder, encoder_path)

# Save labelbinarizer
lb_path = pathlib.Path("model") / "lb.pkl"
save_lb(lb, lb_path)
