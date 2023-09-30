import os
import numpy as np
import pandas as pd
from urllib.parse import urlparse
from pathlib import Path
import joblib
from WineQuality.utils.common import save_json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from WineQuality.entity.config_entity import ModelEvaluationConfig

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

class ModelEvaluation:
    def __init__(self, config = ModelEvaluationConfig):
        self.config = config
    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2
    def log_into_mlflow(self):
        model = joblib.load(self.config.model_path)
        
        X_test = pd.read_csv(self.config.x_test_data_path)
        Y_test = pd.read_csv(self.config.y_test_data_path)
        
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            qualities_pred = model.predict(X_test)
            signature = infer_signature(X_test, qualities_pred)
            (rmse, mae, r2) = self.eval_metrics(Y_test, qualities_pred)
            scores = {
                "rmse":rmse,
                "mae":mae,
                "r2":r2
            }
            save_json(path=Path(self.config.metric_file_name),data=scores)

            mlflow.log_params(self.config.all_params)
            mlflow.log_metric("rmse",rmse)
            mlflow.log_metric("mae",mae)
            mlflow.log_metric("r2",r2)
            mlflow.sklearn.log_model(model, "model")
            # mlflow.sklearn.log_model(
            #     model,
            #     artifact_path = "sklearn-model",
            #     signature = signature,
            #     registered_model_name="sk-learn-elastic-net-model",
            # )
            