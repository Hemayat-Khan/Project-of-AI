import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Simple Streamlit App",
    layout="wide",
    page_icon="📊"
)

st.title("📊 Simple Streamlit Clustering App")
st.write("This is a basic Streamlit application with K-Means clustering.")

# -------------------------
# Upload data
# -------------------------
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv('cleaned_orders.csv')
    st.success("✅ File uploaded successfully!")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # -------------------------
    # Select numeric columns
    # -------------------------
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if len(numeric_cols) < 2:
        st.error("❌ Dataset must contain at least 2 numeric columns.")
    else:
        st.sidebar.header("Clustering Settings")

        selected_features = st.sidebar.multiselect(
            "Select features for clustering",
            numeric_cols,
            default=numeric_cols[:2]
        )

        k = st.sidebar.slider("Number of clusters (k)", 2, 10, 3)

        if len(selected_features) >= 2:
            X = df[selected_features]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            kmeans = KMeans(n_clusters=k, random_state=42)
            df["Cluster"] = kmeans.fit_predict(X_scaled)

            st.subheader("Clustered Data")
            st.dataframe(df.head())

            # -------------------------
            # Visualization
            # -------------------------
            fig = px.scatter(
                df,
                x=selected_features[0],
                y=selected_features[1],
                color="Cluster",
                title="K-Means Clustering Result"
            )
            st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👆 Please upload a CSV file to get started.")
