%%writefile
cleaned_orders_dashboard_ui.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import io
import pickle
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Orders Clustering Studio", layout="wide", page_icon="📦")

# -------------------------
# CSS for beautiful UI
# -------------------------
st.markdown("""
<style>
body { background: linear-gradient(135deg, #f0f4f8 0%, #fdf6f0 100%) !important; }
.block-container { padding-top: 2rem; }
.glass-header {
    backdrop-filter: blur(12px) saturate(160%);
    background: rgba(255, 255, 255, 0.28);
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.3);
    padding: 1.5rem 2rem;
    margin-bottom: 1rem;
    box-shadow: 0 8px 20px rgba(0,0,0,0.1);
}
h1, h2, h3, h4, h5, p, label, span { color: #2d2d2d !important; font-weight: 500 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<div class='glass-header'><h1>📦 Orders Clustering Studio</h1><p>Interactive K-Means clustering dashboard</p></div>",
    unsafe_allow_html=True)

# -------------------------
# Load dataset
# -------------------------
try:
    df = pd.read_csv("cleaned_orders.csv")
    st.success("✅ Dataset loaded successfully!")
except FileNotFoundError:
    st.error("❌ cleaned_orders.csv not found. Please upload your CSV.")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Dataset uploaded successfully!")

# Proceed only if dataset exists
if 'df' in locals():
    st.write("### Dataset Preview", df.head())

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    st.sidebar.header("Clustering Controls")
    k = st.sidebar.slider("Number of clusters (k)", 2, 10, 3)
    log_transform = st.sidebar.checkbox("Apply log-transform to Price_clipped and Order_Total", value=True)

    if log_transform:
        df['Price_clipped_log'] = np.log1p(df['Price_clipped'])
        df['Order_Total_log'] = np.log1p(df['Order_Total'])
        features = ['Quantity', 'Price_clipped_log', 'Order_Total_log']
    else:
        features = numeric_cols

    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit K-Means
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    df['Cluster'] = labels

    # -------------------------
    # Tabs
    # -------------------------
    tabs = st.tabs(["Cluster Info", "PCA Visualization", "Dataset & Export"])

    # ---- Cluster Info ----
    with tabs[0]:
        st.subheader("Cluster Counts & Feature Means")
        st.write("### Cluster Counts")
        st.dataframe(df['Cluster'].value_counts())
        st.write("### Cluster Feature Means")
        st.dataframe(df.groupby('Cluster')[features].mean())

    # ---- PCA Visualization ----
    with tabs[1]:
        st.subheader("2D PCA Scatter Plot of Clusters")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        df['PC1'] = X_pca[:, 0]
        df['PC2'] = X_pca[:, 1]
        fig = px.scatter(df, x='PC1', y='PC2', color='Cluster', title=f'K-Means Clustering (k={k})',
                         color_continuous_scale='Set2', size_max=15)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("Approximate decision regions (optional):")
        xx, yy = np.meshgrid(np.linspace(df['PC1'].min() - 1, df['PC1'].max() + 1, 200),
                             np.linspace(df['PC2'].min() - 1, df['PC2'].max() + 1, 200))
        grid = np.c_[xx.ravel(), yy.ravel()]
        # Map back grid to scaled features space approximately
        inv_X = pca.inverse_transform(grid)
        Z = kmeans.predict(inv_X).reshape(xx.shape)
        fig2 = go.Figure()
        fig2.add_trace(go.Contour(x=np.linspace(df['PC1'].min() - 1, df['PC1'].max() + 1, 200),
                                  y=np.linspace(df['PC2'].min() - 1, df['PC2'].max() + 1, 200),
                                  z=Z, showscale=False, opacity=0.25, contours=dict(showlines=False)))
        fig2.add_trace(go.Scatter(x=df['PC1'], y=df['PC2'], mode='markers',
                                  marker=dict(color=df['Cluster'], colorscale='Set2', size=10),
                                  name='Data Points'))
        st.plotly_chart(fig2, use_container_width=True)

    # ---- Dataset & Export ----
    with tabs[2]:
        st.subheader("Dataset Explorer & Export")
        st.dataframe(df.head())


        # Function to download model
        def model_to_bytes(model):
            bio = io.BytesIO()
            pickle.dump(model, bio)
            bio.seek(0)
            return bio.read()


        st.markdown("Download trained K-Means model (pickle):")
        km_model_bytes = model_to_bytes(kmeans)
        st.download_button("Download K-Means model (.pkl)", data=km_model_bytes, file_name='kmeans_model.pkl',
                           mime='application/octet-stream')

        st.markdown("Download the full clustered dataset:")
        csv_bytes = df.to_csv(index=False).encode()
        st.download_button("Download CSV", data=csv_bytes, file_name='clustered_orders.csv', mime='text/csv')
