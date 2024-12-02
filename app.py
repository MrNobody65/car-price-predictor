from fastapi import FastAPI, HTTPException
from src.pipelines.predict_pipeline import PredictionPipeline, CustomData

app = FastAPI()

prediction_pipeline = PredictionPipeline()

@app.get("/predict/")
def predict(
    manufacture_date: int,
    brand: str,
    model: str,
    origin: str,
    type: str,
    seats: float,
    gearbox: str,
    fuel: str,
    color: str,
    mileage_v2: float,
    condition: str
):
    try:
        custom_data = CustomData(
            manufacture_date=manufacture_date,
            brand_model=brand+" "+model,
            origin=origin,
            type=type,
            seats=seats,
            gearbox=gearbox,
            fuel=fuel,
            color=color,
            mileage_v2=mileage_v2,
            condition=condition,
        )

        input_df = custom_data.get_data_as_dataframe()

        predictions = prediction_pipeline.predict(input_df)

        return {"predictions": predictions.tolist()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
