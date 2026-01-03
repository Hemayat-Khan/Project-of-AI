import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ==============================
# 🌟 PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Attractive Dataset Dashboard",
    layout="wide",
    page_icon="📊"
)

# ==============================
# 🎨 CUSTOM CSS FOR FRONTEND
# ==============================
st.markdown("""
<style>
/* Background Gradient */
body {
    background: linear-gradient(120deg, #f0f8ff, #ffe4e1);
}

/* Card Styles */
.card {
    background: white;
    border-radius: 20px;
    padding: 20px;
    margin: 10px 0;
    box-shadow: 0 8px 20px rgba(0,0,0,0.1);
}

/* Headers */
h1, h2, h3 {
    color: #2c3e50;
    font-family: 'Segoe UI', sans-serif;
}

/* Metrics */
.metric-card {
    background: #ffffffcc;
    border-radius: 15px;
    padding: 20px;
    margin: 10px;
    text-align: center;
    box-shadow: 2px 4px 10px rgba(0,0,0,0.1);
}

.metric-value {
    font-size: 2.5rem;
    font-weight: bold;
    color: #e74c3c;
}

.metric-label {
    font-size: 1.2rem;
    color: #34495e;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #74b9ff, #a29bfe);
    color: #2d3436;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# 📁 DATA UPLOAD
# ==============================
st.title("📊 Attractive Dataset Dashboard")
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv('cleaned_orders.csv')
    st.success("✅ Dataset loaded successfully!")
    
    st.subheader("Dataset Preview")
    st.dataframe(df.head(), height=200)
    
    # ==============================
    # 🌟 Metrics
    # ==============================
    st.subheader("Quick Stats")
    
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    
    if numeric_cols:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Rows</div>
                <div class="metric-value">{df.shape[0]}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Columns</div>
                <div class="metric-value">{df.shape[1]}</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Numeric Columns</div>
                <div class="metric-value">{len(numeric_cols)}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # ==============================
    # 📊 INTERACTIVE VISUALIZATION
    # ==============================
    st.subheader("Interactive Charts")
    
    chart_type = st.selectbox("Select chart type", ["Scatter", "Line", "Bar", "Histogram", "Pie"])
    
    if chart_type in ["Scatter", "Line", "Bar"]:
        x_axis = st.selectbox("X-axis", df.columns)
        y_axis = st.selectbox("Y-axis", df.columns)
        color_col = st.selectbox("Color by (optional)", [None] + df.columns.tolist())
        
        if chart_type == "Scatter":
            fig = px.scatter(df, x=x_axis, y=y_axis, color=color_col, title=f"Scatter: {y_axis} vs {x_axis}")
        elif chart_type == "Line":
            fig = px.line(df, x=x_axis, y=y_axis, color=color_col, title=f"Line: {y_axis} vs {x_axis}")
        elif chart_type == "Bar":
            fig = px.bar(df, x=x_axis, y=y_axis, color=color_col, title=f"Bar: {y_axis} vs {x_axis}")
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Histogram":
        col = st.selectbox("Select column for histogram", numeric_cols)
        fig = px.histogram(df, x=col, nbins=20, title=f"Histogram of {col}")
        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Pie":
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            col = st.selectbox("Select categorical column for pie chart", cat_cols)
            fig = px.pie(df, names=col, title=f"Pie Chart of {col}")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No categorical columns found for pie chart.")
    
    # ==============================
    # 🔍 DATA FILTER
    # ==============================
    st.subheader("Filter Data")
    st.write("Filter your dataset interactively:")
    
    filter_cols = st.multiselect("Select columns to filter", df.columns.tolist(), default=df.columns.tolist()[:3])
    
    filtered_df = df.copy()
    for col in filter_cols:
        unique_vals = df[col].unique().tolist()
        selected_vals = st.multiselect(f"Filter {col}", options=unique_vals, default=unique_vals)
        filtered_df = filtered_df[filtered_df[col].isin(selected_vals)]
    
    st.dataframe(filtered_df, height=300)
    
    # ==============================
    # 💾 DOWNLOAD FILTERED DATA
    # ==============================
    st.markdown("### Download Filtered Dataset")
    csv = filtered_df.to_csv(index=False).encode()
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name="filtered_dataset.csv",
        mime="text/csv"
    )
    
else:
    st.info("👆 Please upload a CSV file to see the attractive dashboard.")
