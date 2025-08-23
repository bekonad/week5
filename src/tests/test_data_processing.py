import pytest
import pandas as pd
from src.data_processing import create_aggregates

def test_create_aggregates():
    sample_data = pd.DataFrame({
        'CustomerId': ['C1', 'C1', 'C2'],
        'Amount': [100, -50, 200]
    })
    agg = create_aggregates(sample_data)
    assert agg.loc[0, 'total_amount'] == 50
    assert agg.loc[0, 'transaction_count'] == 2

def test_missing_handling():
    sample_data = pd.DataFrame({
        'CustomerId': ['C1', 'C1', 'C2'],
        'Amount': [100, None, 200]
    })
    agg = create_aggregates(sample_data)
    assert agg.loc[0, 'transaction_count'] == 2