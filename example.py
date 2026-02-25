import os
import warnings
import sys
import logging
from urllib.parse import urlparse

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import mlflow
import mlflow.sklearn


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    csv_url = "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"

    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception("Error downloading dataset: %s", e)

    train, test = train_test_split(data, test_size=0.25, random_state=42)

    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train["quality"]
    test_y = test["quality"]

    # Tracking URI (AWS Remote Server)
       # Tracking URI (AWS Remote Server)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")


    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    models = {
        "LinearRegression": LinearRegression(),
        "ElasticNet": ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "DecisionTree": DecisionTreeRegressor(random_state=42),
    }

    # =============================
    # REGRESSION MODELS
    # =============================
    for model_name, model in models.items():

        with mlflow.start_run(run_name=model_name):

            model.fit(train_x, train_y)
            preds = model.predict(test_x)

            rmse, mae, r2 = eval_metrics(test_y, preds)

            print(f"\n{model_name} Results")
            print("RMSE:", rmse)
            print("MAE:", mae)
            print("R2:", r2)

            mlflow.log_param("model", model_name)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(
                    model,
                    "model",
                    registered_model_name=f"{model_name}WineModel"
                )
            else:
                mlflow.sklearn.log_model(model, "model")

    # =============================
    # LOGISTIC REGRESSION (Classification)
    # Convert quality into binary class
    # =============================
    train_y_class = (train_y >= 6).astype(int)
    test_y_class = (test_y >= 6).astype(int)

    with mlflow.start_run(run_name="LogisticRegression"):

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_x)
        test_scaled = scaler.transform(test_x)

        log_model = LogisticRegression(max_iter=1000)
        log_model.fit(train_scaled, train_y_class)

        preds = log_model.predict(test_scaled)
        acc = accuracy_score(test_y_class, preds)

        print("\nLogistic Regression Accuracy:", acc)

        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_metric("accuracy", acc)

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                log_model,
                "model",
                registered_model_name="LogisticWineModel"
            )
        else:
            mlflow.sklearn.log_model(log_model, "model")

    # =============================
    # PCA + Linear Regression
    # =============================
    with mlflow.start_run(run_name="PCA_LinearRegression"):

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_x)
        test_scaled = scaler.transform(test_x)

        pca = PCA(n_components=5)
        train_pca = pca.fit_transform(train_scaled)
        test_pca = pca.transform(test_scaled)

        pca_model = LinearRegression()
        pca_model.fit(train_pca, train_y)

        preds = pca_model.predict(test_pca)

        rmse, mae, r2 = eval_metrics(test_y, preds)

        print("\nPCA + Linear Regression Results")
        print("RMSE:", rmse)
        print("MAE:", mae)
        print("R2:", r2)

        mlflow.log_param("model", "PCA_LinearRegression")
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                pca_model,
                "model",
                registered_model_name="PCAWineModel"
            )
        else:
            mlflow.sklearn.log_model(pca_model, "model")
