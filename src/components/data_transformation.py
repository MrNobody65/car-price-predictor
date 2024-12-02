import sys
import os

from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.sparse import issparse

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocess_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer(self):
        try:
            brand_model_pipeline = Pipeline(
                steps=[
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            year_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            origin_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="constant", fill_value="Nước khác")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            type_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="constant", fill_value="Kiểu dáng khác")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            seats_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            gearbox_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            fuel_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            color_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="constant", fill_value="others")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            mileage_v2_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("brand_model_pipeline", brand_model_pipeline, ["brand_model"]),
                    ("year_pipeline", year_pipeline, ["manufacture_date"]),
                    ("origin_pipeline", origin_pipeline, ["origin"]),
                    ("type_pipeline", type_pipeline, ["type"]),
                    ("seats_pipeline", seats_pipeline, ["seats"]),
                    ("gearbox_pipeline", gearbox_pipeline, ["gearbox"]),
                    ("fuel_pipeline", fuel_pipeline, ["fuel"]),
                    ("color_pipeline", color_pipeline, ["color"]),
                    ("mileage_v2_pipeline", mileage_v2_pipeline, ["mileage_v2"])
                ]
            )

            logging.info("Create data transformer")
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, clean_data_path, train_data_path, test_data_path):
        try:
            df = pd.read_csv(clean_data_path)
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info("Read train and test data")

            preprocessor = self.get_data_transformer()
            target_column_name = 'price'

            input = df.drop(columns=[target_column_name], axis=1)

            input_train = train_df.drop(columns=[target_column_name], axis=1)
            target_train = train_df[target_column_name]

            input_test = test_df.drop(columns=[target_column_name], axis=1)
            target_test = test_df[target_column_name]

            preprocessor.fit_transform(input)
            input_train_arr = preprocessor.transform(input_train)
            input_test_arr = preprocessor.transform(input_test)
            
            if issparse(input_train_arr):
                input_train_arr = input_train_arr.toarray()
            if issparse(input_test_arr):
                input_test_arr = input_test_arr.toarray()
            train_arr = np.c_[input_train_arr, np.array(target_train)]
            test_arr = np.c_[np.array(input_test_arr), np.array(target_test)]

            logging.info("Apply transformation to data")

            save_object(
                file_path=self.data_transformation_config.preprocess_obj_file_path,
                obj=preprocessor
            )

            logging.info("Save preprocess object")

            return (
                train_arr,
                test_arr
            )
        except Exception as e:
            raise CustomException(e, sys)