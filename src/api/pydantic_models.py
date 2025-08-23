from pydantic import BaseModel
from typing import List

class CustomerData(BaseModel):
    total_amount: float
    avg_amount: float
    transaction_count: int
    std_amount: float
    avg_hour: float
    avg_day: float
    # Add more features as needed

class PredictionResponse(BaseModel):
    risk_probability: float