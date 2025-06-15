import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# st.set_page_config(page_title="Feature Engineering", layout="wide")
# st.title("🧠 Feature Engineering Insights (Plotly Version)")

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

styled_header("🧠 Feature Engineering")
fade_in_markdown("This section will include feature importance, feature types, and correlation with churn.")








# Load dataset
df = pd.read_csv("data/telco_train.csv")

# Preprocessing
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["SeniorCitizen"] = df["SeniorCitizen"].astype("O")
df.dropna(inplace=True)

# Convert target
if df["churned"].dtype == "O":
    df["churned"] = df["churned"].map({"Yes": 1, "No": 0})

if "customerID" in df.columns:
    df.drop("customerID", axis=1, inplace=True)

# Tabs
tab1, tab2, tab3 = st.tabs([
    "📌 Feature Importance",
    # "🔍 Missing Values",
    "🧾 Feature Types",
    "🧮 Correlation with Churn",
    # "📦 Box Plots"
])

# ---------- TAB 1 ----------
with tab1:
    st.subheader("📌 Feature Importance (Random Forest)")
    df_encoded = df.copy()
    for col in df_encoded.select_dtypes(include="object").columns:
        df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])

    X = df_encoded.drop("churned", axis=1)
    y = df_encoded["churned"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values()
    fig = px.bar(
        x=importances.values,
        y=importances.index,
        orientation="h",
        color=importances.values,
        color_continuous_scale="Sunsetdark",
        title="Random Forest Feature Importances",
        labels={"x": "Importance", "y": "Feature"}
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------- TAB 2 ----------
with tab2:
    st.subheader("💾 Feature Types")
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    num_cols = df.select_dtypes(include=np.number).drop("churned", axis=1).columns.tolist()

    count_df = pd.DataFrame({
        "Type": ["Categorical", "Numerical"],
        "Count": [len(cat_cols), len(num_cols)]
    })

    fig = px.pie(count_df, names="Type", values="Count", title="Categorical vs Numerical Features", hole=0.4)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("🔎 Feature Names"):
        st.markdown(f"**Categorical Features ({len(cat_cols)}):** {', '.join(cat_cols)}")
        st.markdown(f"**Numerical Features ({len(num_cols)}):** {', '.join(num_cols)}")

# ---------- TAB 3 ----------
with tab3:
    st.subheader("🧮 Feature Correlation with Churn")

    df_corr = pd.get_dummies(df, drop_first=True)
    corr = df_corr.corr()["churned"].drop("churned").sort_values()

    corr_df = corr.reset_index()
    corr_df.columns = ["Feature", "Correlation"]

    fig = px.bar(
        corr_df,
        x="Correlation",
        y="Feature",
        orientation="h",
        color="Correlation",
        color_continuous_scale="RdBu",
        title="Correlation of Features with Churn"
    )
    st.plotly_chart(fig, use_container_width=True)