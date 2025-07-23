import os
import sys
import pandas as pd
import numpy as np
import dill
from sklearn.metrics import r2_score, mean_squared_error
from src.logger import logging
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)
        logging.info(f"Object saved at {file_path}")
    except Exception as e:
        raise CustomException(e, sys) from e

def evaluate_model(X_train, Y_train, X_test, Y_test, models):
    try:
        model_report = {}
        for model_name, model in models.items():
            model.fit(X_train, Y_train)
            y_pred = model.predict(X_test)
            model_report[model_name] = {
                "r2_score": r2_score(Y_test, y_pred),
                "mean_squared_error": mean_squared_error(Y_test, y_pred)
            }
        return model_report
    except Exception as e:
        raise CustomException(e, sys) from e