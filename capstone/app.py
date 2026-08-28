from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load('oos_prediction_model.pkl')
model_columns = joblib.load('model_columns.pkl')


class PredictionInput(BaseModel):
    Units_Sold: float
    Units_Ordered: float
    Price: float
    Discount: float
    Promotion: int
    Competitor_Pricing: float
    Epidemic: int
    Demand: float
    Prev_Inventory_Level: float
    Prev_Units_Sold: float
    Rolling_7d_Units_Sold: float
    Day_Of_Week: int
    Month: int
    Category: str
    Region: str
    Weather_Condition: str
    Seasonality: str


@app.post("/predict")
def predict(input_data: PredictionInput):
    input_dict = input_data.dict()

    input_df = pd.DataFrame([input_dict])

    input_df = pd.get_dummies(input_df, columns=['Category', 'Region', 'Weather_Condition', 'Seasonality'])

    for column in model_columns:
        if column not in input_df.columns:
            input_df[column] = 0

    input_df = input_df[model_columns]

    probability = model.predict_proba(input_df)[:, 1][0]

    prediction = int(probability > 0.2)

    return {
        "out_of_stock_probability": float(probability),
        "prediction": prediction
    }