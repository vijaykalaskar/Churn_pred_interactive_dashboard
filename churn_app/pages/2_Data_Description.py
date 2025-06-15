import streamlit as st
import pandas as pd
import os

# ---------- Animated and Styled Header ----------
def fade_in_markdown(text):
    st.markdown(f"""
    <div style="animation: fadeIn 1s ease-in;">
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

# ---------- Page Title ----------
styled_header("📘 Data Description")

# ---------- Load Dataset ----------
data_path = "data/telco_train.csv"
if not os.path.exists(data_path):
    st.error(f"🚫 File not found: {data_path}")
    st.stop()

df = pd.read_csv(data_path)

# ---------- Tabs ----------
tab1, tab2, tab3, tab4 = st.tabs([
    "📄 Data Overview", 
    "🔠 Variable Information", 
    "📊 More Info", 
    "📈 Value Counts + Encoding"
])

# ---------- Tab 1: Overview ----------
with tab1:
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    st.write("📐 Shape of dataset:", df.shape)
    st.write("🧾 Columns:", df.columns.tolist())
    st.write("❓ Missing values:")
    st.dataframe(df.isnull().sum())

    st.markdown("### 📉 Descriptive Statistics:")
    st.dataframe(df.describe())

# ---------- Tab 2: Variable Definitions ----------
with tab2:
    st.markdown("### 🔑 Variable Definitions:")
    st.markdown("""
    - **CustomerId** : Customer ID  
    - **Gender** : Gender (Male, Female)  
    - **SeniorCitizen** : Whether the customer is a senior citizen (1 = Yes, 0 = No)  
    - **Partner** : Whether the client has a partner (Yes, No)  
    - **Dependents** : Whether the client has dependents (Yes, No)  
    - **Tenure** : Number of months the customer has stayed with the company  
    - **PhoneService** : Whether the customer has phone service (Yes, No)  
    - **MultipleLines** : Whether the customer has more than one line  
    - **InternetService** : DSL, Fiber optic, No  
    - **OnlineSecurity** : Yes, No, No Internet service  
    - **OnlineBackup** : Yes, No, No Internet service  
    - **DeviceProtection** : Yes, No, No Internet service  
    - **TechSupport** : Yes, No, No Internet service  
    - **StreamingTV** : Yes, No, No Internet service  
    - **StreamingMovies** : Yes, No, No Internet service  
    - **Contract** : Month-to-month, One year, Two years  
    - **PaperlessBilling** : Yes, No  
    - **PaymentMethod** : Electronic check, Mailed check, Bank transfer, Credit card  
    - **MonthlyCharges** : Amount charged monthly  
    - **TotalCharges** : Total amount charged  
    - **Churn** : 1 = Yes, 0 = No
    """)

# ---------- Tab 3: Dataset Breakdown ----------
with tab3:
    st.markdown("### 📋 Dataset Breakdown:")
    st.markdown("""
    Each row represents a unique customer. The variables contain information related to:

    - **Services**: phone, internet, backup, protection, support, streaming  
    - **Account Info**: tenure, contract, billing, charges  
    - **Demographics**: gender, age, partner, dependents
    """)

# ---------- Tab 4: Value Counts + One-Hot Encoding ----------
with tab4:
    st.subheader("🔁 Value Counts for Categorical Columns")

        # Exclude the first column (e.g., 'customerID') and 'TotalCharges'
    df_subset = df.iloc[:, 1:].drop(columns=['TotalCharges'], errors='ignore')
    # Detect categorical columns
    cat_cols = df_subset.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    if not cat_cols:
        st.info("✅ No categorical columns found.")
    else:
        for col in cat_cols:
            st.markdown(f"#### 🔸 `{col}`")
            st.dataframe(df[col].value_counts(dropna=False).rename_axis(col).reset_index(name='count'))
