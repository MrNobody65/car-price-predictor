import os
import sys

import pickle
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, y_test_pred)
            mse = mean_squared_error(y_test, y_test_pred)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((np.array(y_test) - np.array(y_test_pred)) / np.array(y_test))) * 100
            r2 = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = r2
            logging.info(f"{list(models.keys())[i]} mae: {mae}")
            logging.info(f"{list(models.keys())[i]} mse: {mse}")
            logging.info(f"{list(models.keys())[i]} rmse: {rmse}")
            logging.info(f"{list(models.keys())[i]} mape: {mape}")
            logging.info(f"{list(models.keys())[i]} r2: {r2}")

        return report

    except Exception as e:
        raise CustomException(e, sys)