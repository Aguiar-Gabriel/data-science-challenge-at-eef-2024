import logging

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import max_error, mean_absolute_error, r2_score, mean_squared_error, f1_score, accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import numpy as np


def split_data(data: pd.DataFrame, parameters: dict) -> tuple:
    """Splits data into features and binary target for classification."""
    X = data[parameters["features"]]
    # Transformar o problema em classificação binária: 1 se preço > limiar, 0 caso contrário
    threshold = parameters.get("threshold", 100)  # valor padrão para limiar
    y = (data["price"] > threshold).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"]
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
    """Trains a logistic regression classifier."""
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    return clf


def evaluate_model(
    clf: LogisticRegression, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    """Calculates and logs classification and regression metrics."""
    y_pred = clf.predict(X_test)
    # Métricas de classificação
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    # Métricas de regressão (usando as probabilidades para RMSE/MSE)
    if hasattr(clf, "predict_proba"):
        y_proba = clf.predict_proba(X_test)[:, 1]
    else:
        y_proba = y_pred
    mse = mean_squared_error(y_test, y_proba)
    rmse = mean_squared_error(y_test, y_proba, squared=False)
    mae = mean_absolute_error(y_test, y_proba)
    logger = logging.getLogger(__name__)
    logger.info("Model classification metrics: Accuracy %.3f, F1 %.3f, Recall %.3f, Precision %.3f", acc, f1, recall, precision)
    logger.info("Model regression metrics: MSE %.3f, RMSE %.3f, MAE %.3f", mse, rmse, mae)
    return {"accuracy": acc, "f1": f1, "recall": recall, "precision": precision, "mse": mse, "rmse": rmse, "mae": mae}


# ------------------------------------------------------------------------------------------------------------------
# *project002.ipynb* 
# ------------------------------------------------------------------------------------------------------------------


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    parameters: dict | None = None,
) -> RandomForestClassifier:
    """Train a RandomForestClassifier with SMOTE handling for imbalance.

    Parameters can include ``test_size`` and ``random_state`` keys. If *None*,
    sensible defaults from the original notebook are applied.
    """

    params = parameters or {}
    test_size = params.get("test_size", 0.2)
    random_state = params.get("random_state", 42)

    # ------------------------------------------------------------------
    # Split train/validation following the notebook ratios
    # ------------------------------------------------------------------
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=test_size, stratify=y_train, random_state=random_state
    )

    # ------------------------------------------------------------------
    # Handle class imbalance with SMOTE
    # ------------------------------------------------------------------
    sm = SMOTE(random_state=random_state)
    X_tr_res, y_tr_res = sm.fit_resample(X_tr, y_tr)

    # ------------------------------------------------------------------
    # Train model
    # ------------------------------------------------------------------
    clf = RandomForestClassifier(
        n_estimators=params.get("n_estimators", 100),
        class_weight="balanced",
        random_state=random_state,
    )
    clf.fit(X_tr_res, y_tr_res)

    # ------------------------------------------------------------------
    # Quick validation metric logging (optional)
    # ------------------------------------------------------------------
    logger = logging.getLogger(__name__)
    if not X_val.empty:
        y_pred_val = clf.predict(X_val)
        acc = accuracy_score(y_val, y_pred_val)
        f1 = f1_score(y_val, y_pred_val, zero_division=0)
        logger.info("RandomForest validation – Accuracy: %.3f | F1: %.3f", acc, f1)

    return clf


def predict_espera(
    model: RandomForestClassifier, X_to_predict: pd.DataFrame
) -> np.ndarray:
    """Generate *espera* predictions for the unseen dataset."""

    preds = model.predict(X_to_predict)
    return preds


def build_submission(
    flightid: pd.Series, espera_pred: np.ndarray | pd.Series
) -> pd.DataFrame:
    """Combine flightid and predictions into a DataFrame ready for CSV output."""

    submission = pd.DataFrame({"flightid": flightid, "espera_pred": espera_pred})
    # Ensure order and types match expectations
    submission["espera_pred"] = submission["espera_pred"].astype(float)
    return submission
