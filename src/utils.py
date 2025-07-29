import os
import sys
import pandas as pd
import numpy as np
import dill # to serialize objects
from sklearn.metrics import r2_score, mean_squared_error
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)
        logging.info(f"Object saved at {file_path}")
    except Exception as e:
        raise CustomException(e, sys) from e

def evaluate_model(X_train, Y_train, X_test, Y_test, models, params):
    try:
        model_report = {}
        best_estimators = {}
        for model_name, model in models.items():
            para = params[model_name]
            gs = GridSearchCV(model, para, cv=3, n_jobs=-1, verbose=1)
            gs.fit(X_train, Y_train)

            Y_pred = gs.best_estimator_.predict(X_test)

            r2_square = r2_score(Y_test, Y_pred)
            mse_value = mean_squared_error(Y_test, Y_pred)

            model_report[model_name] = {
                "r2_score": r2_square,
                "mean_squared_error": mse_value
            }
            best_estimators[model_name] = gs.best_estimator_
            logging.info(f"Model: {model_name}, R2 Score: {r2_square}, MSE: {mse_value}")
        return model_report, best_estimators
    except Exception as e:
        raise CustomException(e, sys) from e