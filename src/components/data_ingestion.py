# we just have read the data from csv and spit it into train and test set and then 
# push it into artifacts folder with train.csv, test.csv and data.csv

import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass # to create classes for data attributes
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

#after model trainer.py 
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig



@dataclass # iski wajha se constructor na likhe tho chal jayega 
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion started")
        try:
            # you can change this line wiht your own data source like mongoDB, SQL, API, etc.
            df = pd.read_csv('/Users/vansajrawat/Desktop/5th/mlproject/notebook/data/stud.csv')  # Assuming the data file is in 'data' directory
            logging.info("Data read successfully")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Train and Test sets created")

            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            logging.info("Data Ingestion completed successfully")
            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            raise CustomException(e, sys) from e
            logging.error(f"Error during data ingestion: {e}")


if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_path, test_path = data_ingestion.initiate_data_ingestion()
    print(f"Train data saved at: {train_path}")
    print(f"Test data saved at: {test_path}")

    data_transformation = DataTransformation()
    train_arr, test_arr ,_ = data_transformation.initiate_data_transformation(train_path, test_path)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr,test_arr))
