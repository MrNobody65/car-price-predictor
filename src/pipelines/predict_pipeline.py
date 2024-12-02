import os
import sys
import pandas as pd

from src.exception import CustomException
from src.utils import load_object

class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self, input):
        try:
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            model_path = os.path.join('artifacts', 'model.pkl')
            
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data = preprocessor.transform(input)
            pred = model.predict(data)

            return pred

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
        manufacture_date: int,
        brand_model: str,
        origin: str,
        type: str,
        seats: float,
        gearbox: str,
        fuel: str,
        color: str,
        mileage_v2: float,
        condition: str):
        
        self.manufacture_date = manufacture_date
        self.brand_model = brand_model
        self.origin = origin
        self.type = type
        self.seats = seats
        self.gearbox = gearbox
        self.fuel = fuel
        self.color = color
        self.mileage_v2 = mileage_v2
        self.condition = condition

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "manufacture_date": [self.manufacture_date],
                "brand_model": [self.brand_model],
                "origin": [self.origin],
                "type": [self.type],
                "seats": [self.seats],
                "gearbox": [self.gearbox],
                "fuel": [self.fuel],
                "color": [self.color],
                "mileage_v2": [self.mileage_v2],
                "condition": [self.condition],
            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise CustomException(e, sys)
