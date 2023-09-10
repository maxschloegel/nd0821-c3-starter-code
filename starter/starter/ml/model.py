import pickle

from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier

from .data import process_data
from .config import cat_features, label


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    # read RFC-parameters from config file?
    rfc_config = []
    model = RandomForestClassifier(n_estimators=50, max_depth=20, min_samples_split=5) 
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def compute_model_metrics_for_feature_slice(model, df, categorical_features, label, slice_column, encoder, lb):
    """Computes metrics for all slices defined by slice column

    This function iterates over all classes in `slice_column` and will output 
    the metrics for all uniqe column features

    Inputs
    ------
    df : pandas.DataFrame
        contains data to be slices
    predictors: List[str]
        List of predicter variable names



    Returns
    -------
    metric_dict: dict
        Dictionary mapping unique feature name to metrics.
            i.e.: {unique_feature: [precision, recall, fbeta]}
    """
    metric_dict = {}

    for value in df[slice_column].unique():
        df_temp = df[df[slice_column] == value]
        X, y, _, _ = process_data(df_temp, categorical_features=categorical_features, label=label, training=False, encoder=encoder, lb=lb)
        y_pred = inference(model, X)
        metrics = compute_model_metrics(y, y_pred)
        metric_dict[value] = metrics

    return metric_dict



def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def save_model(model, path):
    """ Save model

    Inputs
    ------
    model : ???
        Trained machine learning model
    path : str|pathlib.Path
        Path where the model is saved to
    """
    with open(path, "wb") as f:
        pickle.dump(model, f)


def save_encoder(encoder, path):
    """Save encoder

    Inputs
    ------
    encoder : sklearn.proprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder for categorical features
    path : str|pathlib.Path
        Path where the encoder is saved to
    """
    with open(path, "wb") as f:
        pickle.dump(encoder, f)
