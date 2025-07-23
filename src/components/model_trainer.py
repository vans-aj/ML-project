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

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path): # for model training
        try:
            logging.info("Model Training started")
            logging.info("splitting train and test arrays into features and target variable")
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
                "CatBoost": CatBoostRegressor()
            }
            model_report: dict = evaluate_model(X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test, models=models)

            best_model_score = max(sorted(model_report.values(), key=lambda x: x['r2_score']))
            best_model_name = [name for name, score in model_report.items() if score == best_model_score][0]
            best_model = models[best_model_name]

        except:
            pass