import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
from data_processing import process_data, preprocess_pipeline
import numpy as np

def evaluate_model(y_true, y_pred, y_prob):
    # Handle case where y_prob might be None or only one class present
    if len(np.unique(y_true)) == 1:
        roc_auc = 0.5  # Default value for single class
    else:
        roc_auc = roc_auc_score(y_true, y_prob)
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc
    }

def train_models():
    try:
        # Process data with error handling
        df, preprocessor = process_data('../data/raw/data.csv', '../data/processed/processed.csv')
        
        # Check if required columns exist
        if 'CustomerId' not in df.columns or 'is_high_risk' not in df.columns:
            raise ValueError("Required columns (CustomerId, is_high_risk) not found in dataset")
        
        X = df.drop(['CustomerId', 'is_high_risk'], axis=1)
        y = df['is_high_risk']

        # Split data first to avoid leakage
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Fit and transform using the preprocessor
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)

        # Get feature names dynamically
        try:
            feature_names = preprocessor.get_feature_names_out()
        except AttributeError:
            # Fallback for older sklearn versions
            feature_names = [f'feature_{i}' for i in range(X_train_transformed.shape[1])]
        
        input_example = pd.DataFrame(X_train_transformed[:1], columns=feature_names)

        mlflow.set_experiment("CreditRisk")
        best_model = None
        best_roc_auc = 0
        best_model_name = None

        with mlflow.start_run() as run:
            run_id = run.info.run_id

            # Logistic Regression
            lr = LogisticRegression(random_state=42, max_iter=1000)
            lr_params = {'C': [0.1, 1, 10]}
            lr_grid = GridSearchCV(lr, lr_params, cv=5, scoring='roc_auc')
            lr_grid.fit(X_train_transformed, y_train)
            lr_pred = lr_grid.predict(X_test_transformed)
            lr_prob = lr_grid.predict_proba(X_test_transformed)[:, 1]
            lr_metrics = evaluate_model(y_test, lr_pred, lr_prob)
            
            mlflow.log_params(lr_grid.best_params_)
            mlflow.log_metrics(lr_metrics)
            mlflow.sklearn.log_model(
                lr_grid.best_estimator_,
                "logistic_regression",
                input_example=input_example
            )

            # Random Forest
            rf = RandomForestClassifier(random_state=42)
            rf_params = {'n_estimators': [50, 100], 'max_depth': [None, 10]}
            rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='roc_auc')
            rf_grid.fit(X_train_transformed, y_train)
            rf_pred = rf_grid.predict(X_test_transformed)
            rf_prob = rf_grid.predict_proba(X_test_transformed)[:, 1]
            rf_metrics = evaluate_model(y_test, rf_pred, rf_prob)
            
            mlflow.log_params(rf_grid.best_params_)
            mlflow.log_metrics(rf_metrics)
            mlflow.sklearn.log_model(
                rf_grid.best_estimator_,
                "random_forest",
                input_example=input_example
            )

            # Compare models and register the best one
            if rf_metrics['roc_auc'] > lr_metrics['roc_auc']:
                best_model = rf_grid.best_estimator_
                best_roc_auc = rf_metrics['roc_auc']
                best_model_name = "random_forest"
            else:
                best_model = lr_grid.best_estimator_
                best_roc_auc = lr_metrics['roc_auc']
                best_model_name = "logistic_regression"

            mlflow.log_metric('best_model_roc_auc', best_roc_auc)
            mlflow.set_tag('best_model', best_model_name)
            
            # Register best model
            mlflow.register_model(f"runs:/{run_id}/{best_model_name}", "BestModel")

        return run_id

    except Exception as e:
        print(f"Error in train_models: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        run_id = train_models()
        print(f"MLflow Run ID: {run_id}")
    except Exception as e:
        print(f"Failed to train models: {str(e)}")
        exit(1)
