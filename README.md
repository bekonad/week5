# Credit Risk Model

## Credit Scoring Business Understanding

- **How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?**  
  The Basel II Accord requires banks to maintain capital reserves based on precise credit risk assessments, necessitating interpretable models (e.g., Logistic Regression with WoE) for regulatory compliance and auditability. Thorough documentation ensures transparency for regulators and stakeholders.

- **Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?**  
  Without a direct default label, a proxy variable (e.g., RFM-based clustering) is needed to estimate credit risk. Risks include misclassification if the proxy poorly correlates with actual defaults, leading to incorrect loan approvals, increased bad debt, or lost opportunities from denying creditworthy customers.

- **What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?**  
  Simple models are interpretable, easier to justify under regulations, and computationally efficient but may have lower accuracy. Complex models offer better predictive performance and handle non-linear patterns but are less interpretable, posing challenges for regulatory approval and increasing overfitting risks.

## Setup Instructions
1. Place `data.csv` in `data/raw/`.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run data processing: `python src/data_processing.py`.
4. Train models: `python src/train.py`.
5. Run API: `uvicorn src.api.main:app --host 0.0.0.0 --port 8000`.
6. View MLflow UI: `mlflow ui`.

## EDA
See `notebooks/1.0-eda.ipynb` for exploratory data analysis and insights.
