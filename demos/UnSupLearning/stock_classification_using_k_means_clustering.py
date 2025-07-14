# Install required packages if missing
import subprocess, sys
for pkg in ['yfinance', 'pandas', 'numpy', 'matplotlib', 'plotly', 'scipy', 'scikit-learn']:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg])

import numpy as np
import pandas as pd
import yfinance as yf
from math import sqrt
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from scipy.cluster.vq import vq
from sklearn.cluster import KMeans

# Fetch S&P 500 tickers
sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
data_table = pd.read_html(sp500_url)
tickers = data_table[0]['Symbol'].str.replace(r'\.','-', regex=True).tolist()  # clean symbols :contentReference[oaicite:0]{index=0}

# Download adjusted close prices for the past year
prices_list = []
for ticker in tickers:
    try:
        df = yf.Ticker(ticker).history(
            start='2023-06-13', end='2024-06-13',
            auto_adjust=True, actions=True
        )[['Close']].rename(columns={'Close': ticker})
        prices_list.append(df)
    except Exception:
        # skip tickers that fail
        pass

# Concatenate once, after the loop
prices_df = pd.concat(prices_list, axis=1).sort_index()

# Compute annualized return and volatility
returns = pd.DataFrame({
    'Returns': prices_df.pct_change().mean() * 252,
    'Volatility': prices_df.pct_change().std() * sqrt(252)
})

# Prepare data for clustering
X = returns[['Returns', 'Volatility']].to_numpy()

# Determine optimal k via elbow method
sse = []
for k in range(1, 20):
    km = KMeans(n_clusters=k, random_state=42).fit(X)
    sse.append(km.inertia_)
plt.figure()
plt.plot(range(1, 20), sse, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('SSE')
plt.show()

# Fit final model
K = 7
kmeans = KMeans(n_clusters=K, random_state=42).fit(X)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# Build results DataFrame
clusters_df = returns.reset_index().rename(columns={'index':'Ticker'})
clusters_df['Cluster'] = labels
centroids_df = pd.DataFrame(centroids, columns=['Returns','Volatility'])
centroids_df['Cluster'] = range(K)

# Plot clusters with centroids
fig = px.scatter(
    clusters_df, x='Returns', y='Volatility',
    color='Cluster', hover_data=['Ticker']
)
fig.add_trace(go.Scatter(
    x=centroids_df['Returns'], y=centroids_df['Volatility'],
    mode='markers+text', marker=dict(color='black', size=15, symbol='x'),
    text=centroids_df['Cluster'], textposition='top center', name='Centroids'
))
fig.update_layout(coloraxis_showscale=False, height=800)
fig.show()

# Summary
cluster_summary = clusters_df.groupby('Cluster').agg(
    Avg_Return=('Returns','mean'),
    Avg_Volatility=('Volatility','mean'),
    Num_Tickers=('Ticker','count')
).reset_index()
cluster_summary['Return_Volatility_Ratio'] = (
    cluster_summary['Avg_Return'] / cluster_summary['Avg_Volatility']
)

print(cluster_summary)
for c in cluster_summary['Cluster']:
    print(f"Tickers in cluster {c}:",
          clusters_df[clusters_df['Cluster']==c]['Ticker'].tolist())
