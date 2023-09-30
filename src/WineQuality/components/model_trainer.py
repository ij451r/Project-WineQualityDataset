import os
from WineQuality import logger
import pandas as pd
from sklearn.linear_model import ElasticNet
import joblib
from WineQuality.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config = ModelTrainerConfig):
        self.config = config

    def train(self):
        X_train = pd.read_csv(self.config.x_train_data_path)
        Y_train = pd.read_csv(self.config.y_train_data_path)

        model = ElasticNet(alpha = self.config.alpha, l1_ratio = self.config.l1_ratio, random_state=42)
        model.fit(X_train, Y_train)

        joblib.dump(model, os.path.join(self.config.root_dir, self.config.model_name))
        logger.info(f"Model: {self.config.model_name} trained and saved in {self.config.root_dir}")