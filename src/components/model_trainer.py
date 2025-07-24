import os
import sys
from dataclasses import dataclass # to create classes for data attributes

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig: # to create a class for model training configuration
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array): # for model training
        try:
            logging.info("Model Training started")
            logging.info("splitting train and test arrays into features and target variable")
            logging.info(f"Type of train_array: {type(train_array)}")
            logging.info(f"Type of test_array: {type(test_array)}")
            import numpy as np

            # Ensure the arrays are numpy arrays
            train_array = np.array(train_array)
            test_array = np.array(test_array)
            logging.info(f"train_array shape: {train_array.shape}")
            logging.info(f"test_array shape: {test_array.shape}")

            X_train, Y_train = train_array[:, :-1], train_array[:, -1] # train arrya and test array are receving from data transformation.py
            X_test, Y_test = test_array[:, :-1], test_array[:, -1]
            models = {
                "RandomForest": RandomForestRegressor(),
                "GradientBoosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "LinearRegression": LinearRegression(),
                "KNeighbors": KNeighborsRegressor(),
                "XGBoost": XGBRegressor(),
                "DecisionTree": DecisionTreeRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False)
            }
            model_report: dict = evaluate_model(X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test, models=models)

            # Find the best model by r2_score
            best_model_name = max(model_report, key=lambda name: model_report[name]['r2_score'])
            best_model_score = model_report[best_model_name]['r2_score']
            best_model = models[best_model_name]
            if (best_model_score < 0.6):
                raise CustomException("No best model found with sufficient accuracy", sys)
            logging.info(f"Best model found: {best_model_name} with R2 score: {best_model_score}")

            save_object( # save the best model to a file
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info("Model saved ")
            
            predict = best_model.predict(X_test)
            r2_square = r2_score(Y_test, predict)

            return r2_square

        except Exception as e:
            logging.error(f"Error occurred during model training: {e}")
            raise CustomException("Model training failed", sys)