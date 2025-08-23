from fastapi import FastAPI
import mlflow.sklearn
from pydantic_models import CustomerData, PredictionResponse
from src.data_processing import preprocess_pipeline

app = FastAPI()

model = mlflow.sklearn.load_model("models:/BestCreditRiskModel/Production")
preprocessor = preprocess_pipeline()

@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerData):
    df = pd.DataFrame([data.dict()])
    processed = preprocessor.transform(df)
    prob = model.predict_proba(processed)[:, 1][0]
    return {"risk_probability": prob}