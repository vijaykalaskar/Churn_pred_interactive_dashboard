import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Animations + headers
def fade_in_markdown(text):
    st.markdown(f"""
    <div style="animation: fadeIn 1s ease-in; -webkit-animation: fadeIn 1s ease-in;">
        {text}
    </div>
    <style>
    @keyframes fadeIn {{
        from {{ opacity: 0; }}
        to {{ opacity: 1; }}
    }}
    </style>
    """, unsafe_allow_html=True)

def styled_header(title):
    fade_in_markdown(f"<h2 style='font-size: 36px; font-weight: bold; color: #1f77b4;'>{title}</h2>")

styled_header("📊 Data Visualization")
fade_in_markdown("This section will include churn distribution, and correlation heatmaps.")

# Load data
df = pd.read_csv("data/telco_train.csv")

# Clean/convert
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df["SeniorCitizen"] = df["SeniorCitizen"].astype("O")
if df["churned"].dtype == "O":
    df["churned"] = df["churned"].map({"Yes": 1, "No": 0})

# Define 7 tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Churn by Categories",
    "📈 Numeric Trends",
    "💠 Correlation Heatmap",
    "🧾 Box Plots of Numerical Features by Churn",
    # "📌 Feature Importance (Random Forest)",
    # "🧾 Feature Types",
    # "🧮 Feature Correlation with Churn"
])

# ---------- TAB 1 ----------
with tab1:
    st.subheader("🔁 Churn Distribution by Categorical Columns")
    categorical_cols = ["MultipleLines", "InternetService", "Contract", "PaymentMethod", "OnlineSecurity"]

    for col in categorical_cols:
        if col in df.columns:
            temp_df = df[[col, "churned"]].copy()
            temp_df = temp_df.value_counts().reset_index(name="count")
            temp_df.columns = [col, "Churn", "count"]
            temp_df["Churn"] = temp_df["Churn"].astype(str)
            total_per_category = temp_df.groupby(col)["count"].transform("sum")
            temp_df["percent"] = round((temp_df["count"] / total_per_category) * 100, 2)

            fig = px.bar(
                temp_df,
                x=col,
                y="count",
                color="Churn",
                barmode="group",
                text="count",
                hover_data={col: True, "Churn": True, "count": True, "percent": True},
                title=f"{col} vs Churn",
                labels={"count": "Customer Count", "percent": "% within group"},
                color_discrete_map={"0": "lightskyblue", "1": "crimson"}
            )
            fig.update_layout(
                xaxis_title=col,
                yaxis_title="Customer Count",
                legend_title="Churned",
                uniformtext_minsize=8,
                uniformtext_mode='hide'
            )
            st.plotly_chart(fig, use_container_width=True)

# ---------- TAB 2 ----------
with tab2:
    st.subheader("📉 Distribution of Numeric Columns by Churn")
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    for col in numeric_cols:
        fig = px.histogram(df, x=col, color="churned", nbins=50, barmode="overlay",
                           color_discrete_map={0: "lightskyblue", 1: "crimson"},
                           title=f"Distribution of {col} by Churn")
        fig.update_layout(xaxis_title=col, yaxis_title="Count", legend_title="Churned")
        st.plotly_chart(fig, use_container_width=True)

# ---------- TAB 3 ----------
with tab3:
    st.subheader("🔗 Correlation of Features with Churn")
    cols_to_include = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
        "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges", "churned"
    ]
    df_corr = df[cols_to_include].copy()
    df_encoded = pd.get_dummies(df_corr.drop(columns=["churned"]), drop_first=True)
    df_encoded["churned"] = df_corr["churned"]

    correlations = df_encoded.corr()["churned"].drop("churned").sort_values(ascending=False)
    fig = px.bar(
        x=correlations.values,
        y=correlations.index,
        orientation='h',
        title="Correlation of Features with Churn",
        labels={"x": "Correlation with churned", "y": "Features"},
        color=correlations.values,
        color_continuous_scale="RdBu",
        text=correlations.round(2)
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=800)
    st.plotly_chart(fig, use_container_width=True)

    # Sub heatmap
    st.subheader("📊 Correlation Heatmap: Tenure, Charges & Churn")
    selected_cols = ["tenure", "MonthlyCharges", "TotalCharges", "churned"]
    corr_subset = df[selected_cols].corr().round(2)
    heatmap_fig = px.imshow(
        corr_subset,
        text_auto=True,
        color_continuous_scale="RdBu",
        title="Correlation Among Tenure, Charges, and Churn",
        labels={"color": "Correlation"},
        aspect="auto"
    )
    heatmap_fig.update_layout(width=700, height=600)
    st.plotly_chart(heatmap_fig, use_container_width=True)

# ---------- TAB 4 ----------
with tab4:
    st.subheader("📦 Box Plots of Numerical Features by Churn")
    st.markdown("""
    **🔍 What is a Box Plot?**

    A box plot shows:
    - **Median**
    - **Interquartile range (IQR)**
    - **Whiskers (min/max)**
    - **Outliers**

    Helps you compare **distributions** between churned vs non-churned.
    """)
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    for col in numeric_cols:
        fig = px.box(
            df,
            x="churned",
            y=col,
            points="all",
            color="churned",
            color_discrete_map={0: "lightskyblue", 1: "crimson"},
            title=f"📦 {col} Distribution by Churn",
            labels={"churned": "Churned", col: col}
        )
        fig.update_layout(xaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=["No", "Yes"]))
        st.plotly_chart(fig, use_container_width=True)

