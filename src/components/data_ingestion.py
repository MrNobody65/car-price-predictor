import os
import sys

from src.exception import CustomException
from src.logger import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    clean_data_path: str = os.path.join('artifacts', 'data.csv')
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            df = pd.read_csv('data/car.csv')
            df.drop(["id", "list_id", "list_time"], axis=1, inplace=True)
            brand_dis = df["brand"].value_counts(dropna=False, normalize=True).cumsum()
            last_index = brand_dis[brand_dis >= 0.9].index[0]
            brands = brand_dis[brand_dis <= 0.9].index.to_list()
            brands.append(last_index)
            df = df[df["brand"].isin(brands)]
            df.dropna(subset=["price"], inplace=True)
            df["brand_model"] = df["brand"] + " " + df["model"]
            df.drop(["brand", "model"], axis=1, inplace=True)
            df["seats"].replace(-1, np.nan)

            df.to_csv(self.ingestion_config.clean_data_path, index=False, header=True)

            logging.info('Read the dataset as dataframe and process')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            used_df = df[df.condition == 'used']
            new_df = df[df.condition == 'new']

            logging.info('Split the dataset based on condition')

            train_used_set, test_used_set = train_test_split(used_df, test_size=0.2, random_state=42)
            train_new_set, test_new_set = train_test_split(new_df, test_size=0.2, random_state=42)

            train_set = pd.concat([train_used_set, train_new_set], ignore_index=True)
            test_set = pd.concat([test_used_set, test_new_set], ignore_index=True)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion is completed")

            return (
                self.ingestion_config.clean_data_path,
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        
        except Exception as e:
            raise CustomException(e, sys)