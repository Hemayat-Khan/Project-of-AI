import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Restaurant Performance Dashboard",
    page_icon="📊",
    layout="wide"
)

# ==============================
# CLEAN PROFESSIONAL THEME
# ==============================
st.markdown("""
<style>
body {
    background-color: #f5f6fa;
}

h1, h2, h3 {
    color: #2f3640;
}

.kpi-card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
    text-align: center;
}

.kpi-value {
    font-size: 2.2rem;
    font-weight: bold;
    color: #273c75;
}

.kpi-label {
    font-size: 1rem;
    color: #718093;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# REALISTIC RESTAURANT DATA
# ==============================
@st.cache_data
def load_data():
    dates = pd.date_range(
        start=datetime.today() - timedelta(days=30),
        periods=30
    )

    data = {
        "Date": dates,
        "Orders": [120 + i % 15 for i in range(30)],
        "Revenue": [2500 + i * 45 for i in range(30)],
        "Cost": [1500 + i * 30 for i in range(30)]
    }

    df = pd.DataFrame(data)
    df["Profit"] = df["Revenue"] - df["Cost"]
    df["AOV"] = df["Revenue"] / df["Orders"]

    return df

df = load_data()

# ==============================
# HEADER
# ==============================
st.title("📊 Restaurant Performance Dashboard")
st.write("Operational and financial overview of restaurant performance")

# ==============================
# DATE FILTER
# ==============================
with st.sidebar:
    st.header("Filters")
    start_date = st.date_input("Start Date", df["Date"].min())
    end_date = st.date_input("End Date", df["Date"].max())

filtered_df = df[
    (df["Date"] >= pd.to_datetime(start_date)) &
    (df["Date"] <= pd.to_datetime(end_date))
]

# ==============================
# KPI SECTION
# ==============================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Total Revenue</div>
        <div class="kpi-value">${filtered_df['Revenue'].sum():,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Total Cost</div>
        <div class="kpi-value">${filtered_df['Cost'].sum():,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Total Profit</div>
        <div class="kpi-value">${filtered_df['Profit'].sum():,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Avg Order Value</div>
        <div class="kpi-value">${filtered_df['AOV'].mean():.2f}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ==============================
# REVENUE & PROFIT TREND
# ==============================
st.subheader("Revenue & Profit Trend")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=filtered_df["Date"],
    y=filtered_df["Revenue"],
    name="Revenue",
    line=dict(width=3)
))
fig.add_trace(go.Scatter(
    x=filtered_df["Date"],
    y=filtered_df["Profit"],
    name="Profit",
    line=dict(width=3)
))

fig.update_layout(
    height=400,
    xaxis_title="Date",
    yaxis_title="Amount ($)",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

# ==============================
# ORDERS BAR CHART
# ==============================
st.subheader("Daily Orders")

fig2 = px.bar(
    filtered_df,
    x="Date",
    y="Orders",
    title="Orders per Day"
)

fig2.update_layout(height=350)

st.plotly_chart(fig2, use_container_width=True)

# ==============================
# DATA TABLE
# ==============================
st.subheader("Detailed Performance Table")

st.dataframe(
    filtered_df.style.format({
        "Revenue": "${:,.0f}",
        "Cost": "${:,.0f}",
        "Profit": "${:,.0f}",
        "AOV": "${:.2f}"
    }),
    use_container_width=True
)

# ==============================
# FOOTER
# ==============================
st.markdown(
    "<center style='color:gray;'>Restaurant Analytics Dashboard • Real-World Business View</center>",
    unsafe_allow_html=True
)
