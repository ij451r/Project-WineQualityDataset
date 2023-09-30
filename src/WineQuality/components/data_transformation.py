import os
from WineQuality import logger
import pandas as pd
from WineQuality.entity.config_entity import DataTransformationConfig
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataTransformation:
    def __init__(self, config = DataTransformationConfig):
        self.config = config
        self.df = pd.read_csv(self.config.data_path)

    def drop_non_validated_columns(self):
        with open(self.config.STATUS_FILE, 'r') as status_file:
            for validation in status_file:
                if validation.split(' ')[2] == 'False':
                    column = ' '.join(validation.split(' ')[5:]).rstrip() 
                    self.df.drop(columns=[column], inplace=True)
                    logger.info(f"Dropped Column {column} as it did not pass validation")
    
    def train_test_split(self):
        scaler = StandardScaler()
        X = self.df.drop(columns=[self.config.target_column], axis=1)
        Y = self.df[[self.config.target_column]]

        X_train, X_test, Y_train, Y_test = train_test_split(X , Y, test_size=0.2, random_state=22)
        X_train.to_csv(os.path.join(self.config.root_dir,"X_train.csv" ), index = False)
        X_test.to_csv(os.path.join(self.config.root_dir,"X_test.csv"), index = False)
        Y_train.to_csv(os.path.join(self.config.root_dir,"Y_train.csv"), index = False)
        Y_test.to_csv(os.path.join(self.config.root_dir,"Y_test.csv"), index = False)

        logger.info(f"Data splitted into X_train, X_test, Y_train, Y_test")
        logger.info(f"X_train.shape: {X_train.shape}")
        logger.info(f"X_test.shape: {X_test.shape}")
        logger.info(f"Y_train.shape: {Y_train.shape}")
        logger.info(f"Y_test.shape: {Y_test.shape}")
        
        