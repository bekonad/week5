import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from xverse.transformer import WOE
from datetime import datetime

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    return df

def create_aggregates(df):
    agg_df = df.groupby('CustomerId').agg(
        total_amount=('Amount', 'sum'),
        avg_amount=('Amount', 'mean'),
        transaction_count=('Amount', 'count'),
        std_amount=('Amount', 'std')
    ).reset_index()
    return agg_df

def extract_time_features(df):
    df['transaction_hour'] = df['TransactionStartTime'].dt.hour
    df['transaction_day'] = df['TransactionStartTime'].dt.day
    df['transaction_month'] = df['TransactionStartTime'].dt.month
    df['transaction_year'] = df['TransactionStartTime'].dt.year
    time_agg = df.groupby('CustomerId').agg(
        avg_hour=('transaction_hour', 'mean'),
        avg_day=('transaction_day', 'mean')
    ).reset_index()
    return time_agg

def calculate_rfm(df):
    snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('CustomerId').agg(
        Recency=('TransactionStartTime', lambda x: (snapshot_date - x.max()).days),
        Frequency=('TransactionId', 'count'),
        Monetary=('Amount', 'sum')
    ).reset_index()
    rfm['Monetary'] = rfm['Monetary'].abs()
    return rfm

def create_proxy_target(rfm):
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    kmeans = KMeans(n_clusters=3, random_state=42)
    rfm['cluster'] = kmeans.fit_predict(rfm_scaled)
    cluster_means = rfm.groupby('cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    high_risk_cluster = cluster_means[cluster_means['Recency'] == cluster_means['Recency'].max()].index[0]
    rfm['is_high_risk'] = (rfm['cluster'] == high_risk_cluster).astype(int)
    return rfm[['CustomerId', 'is_high_risk']]

def preprocess_pipeline():
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, ['total_amount', 'avg_amount', 'transaction_count', 'std_amount', 'avg_hour', 'avg_day']),
        ])
    return preprocessor

def process_data(input_path, output_path):
    df = load_data(input_path)
    agg_df = create_aggregates(df)
    time_df = extract_time_features(df)
    rfm = calculate_rfm(df)
    target_df = create_proxy_target(rfm)
    processed_df = agg_df.merge(time_df, on='CustomerId').merge(target_df, on='CustomerId')
    processed_df.to_csv(output_path, index=False)
    return processed_df, preprocessor

if __name__ == "__main__":
    process_data('../data/raw/data.csv', '../data/processed/processed.csv')