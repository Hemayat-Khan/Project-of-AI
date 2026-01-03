import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import io
import pickle
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Orders Clustering Studio", layout="wide", page_icon="📦")

# -------------------------
# CSS
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
}
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
    st.error("❌ cleaned_orders.csv not found.")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Dataset uploaded successfully!")

if 'df' in locals():
    st.write("### Dataset Preview", df.head())

    st.sidebar.header("Clustering Controls")
    k = st.sidebar.slider("Number of clusters (k)", 2, 10, 3)
    log_transform = st.sidebar.checkbox("Apply log-transform", value=True)

    if log_transform:
        df['Price_clipped_log'] = np.log1p(df['Price_clipped'])
        df['Order_Total_log'] = np.log1p(df['Order_Total'])
        features = ['Quantity', 'Price_clipped_log', 'Order_Total_log']
    else:
        features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    tabs = st.tabs(["Cluster Info", "PCA Visualization", "Dataset & Export"])

    # ---- Cluster Info ----
    with tabs[0]:
        st.subheader("Cluster Counts")
        st.dataframe(df['Cluster'].value_counts())

        st.subheader("Cluster Feature Means")
        st.dataframe(df.groupby('Cluster')[features].mean())

    # ---- PCA Visualization ----
    with tabs[1]:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        df['PC1'], df['PC2'] = X_pca[:, 0], X_pca[:, 1]

        fig = px.scatter(
            df, x='PC1', y='PC2', color='Cluster',
            title=f'K-Means Clustering (k={k})'
        )
        st.plotly_chart(fig, use_container_width=True)

        # ---- Decision regions (FIXED) ----
        xx, yy = np.meshgrid(
            np.linspace(df['PC1'].min()-1, df['PC1'].max()+1, 200),
            np.linspace(df['PC2'].min()-1, df['PC2'].max()+1, 200)
        )

        grid = np.c_[xx.ravel(), yy.ravel()]
        inv_features = pca.inverse_transform(grid)
        inv_scaled = scaler.transform(inv_features)  # ✅ FIX
        Z = kmeans.predict(inv_scaled).reshape(xx.shape)

        fig2 = go.Figure()
        fig2.add_trace(go.Contour(z=Z, showscale=False, opacity=0.3))
        fig2.add_trace(go.Scatter(
            x=df['PC1'], y=df['PC2'], mode='markers',
            marker=dict(color=df['Cluster'], size=8)
        ))
        st.plotly_chart(fig2, use_container_width=True)

    # ---- Export ----
    with tabs[2]:
        def model_to_bytes(model):
            bio = io.BytesIO()
            pickle.dump(model, bio)
            bio.seek(0)
            return bio.read()

        st.download_button(
            "Download K-Means model",
            data=model_to_bytes(kmeans),
            file_name="kmeans_model.pkl"
        )

        st.download_button(
            "Download Clustered CSV",
            data=df.to_csv(index=False).encode(),
            file_name="clustered_orders.csv"
        )
