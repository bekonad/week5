import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
from data_processing import process_data, preprocess_pipeline

def evaluate_model(y_true, y_pred, y_prob):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_prob)
    }

def train_models():
    # Process data
    df, preprocessor = process_data('../data/raw/data.csv', '../data/processed/processed.csv')
    X = df.drop(['CustomerId', 'is_high_risk'], axis=1)
    y = df['is_high_risk']
    X = preprocessor.fit_transform(X)

    # Create input example for MLflow
    input_example = pd.DataFrame(
        X[:1],
        columns=['total_amount', 'avg_amount', 'transaction_count', 'std_amount', 'avg_hour', 'avg_day']
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_experiment("CreditRisk")
    with mlflow.start_run() as run:
        run_id = run.info.run_id

        # Logistic Regression
        lr = LogisticRegression()
        lr_params = {'C': [0.1, 1, 10]}
        lr_grid = GridSearchCV(lr, lr_params, cv=5)
        lr_grid.fit(X_train, y_train)
        lr_pred = lr_grid.predict(X_test)
        lr_prob = lr_grid.predict_proba(X_test)[:, 1]
        lr_metrics = evaluate_model(y_test, lr_pred, lr_prob)
        mlflow.log_params(lr_grid.best_params_)
        mlflow.log_metrics(lr_metrics)
        mlflow.sklearn.log_model(
            lr_grid.best_estimator_,
            "logistic_regression",
            input_example=input_example
        )

        # Random Forest
        rf = RandomForestClassifier()
        rf_params = {'n_estimators': [50, 100], 'max_depth': [None, 10]}
        rf_grid = GridSearchCV(rf, rf_params, cv=5)
        rf_grid.fit(X_train, y_train)
        rf_pred = rf_grid.predict(X_test)
        rf_prob = rf_grid.predict_proba(X_test)[:, 1]
        rf_metrics = evaluate_model(y_test, rf_pred, rf_prob)
        mlflow.log_params(rf_grid.best_params_)
        mlflow.log_metrics(rf_metrics)
        mlflow.sklearn.log_model(
            rf_grid.best_estimator_,
            "random_forest",
            input_example=input_example
        )

        # Register best model (assume Random Forest)
        mlflow.register_model(f"runs:/{run_id}/random_forest", "BestModel")

    return run_id


if __name__ == "__main__":
    run_id = train_models()
    print(f"MLflow Run ID: {run_id}")