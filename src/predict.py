import mlflow.sklearn
import pandas as pd
from data_processing import preprocess_pipeline

def predict(new_data, run_id):
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/random_forest")
    preprocessor = preprocess_pipeline()
    new_data_processed = preprocessor.transform(new_data)
    probs = model.predict_proba(new_data_processed)[:, 1]
    return probs

# Example usage:
# df = pd.DataFrame({...})
# probs = predict(df, "your_run_id")