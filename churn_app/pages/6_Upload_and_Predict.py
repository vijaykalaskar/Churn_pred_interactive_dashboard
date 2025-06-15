import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
import plotly.graph_objs as go

# ----------- Utility Functions -----------

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

def preprocess_input(df, cat_cols, num_cols, scaler):
    df = df.copy()
    df[num_cols] = scaler.transform(df[num_cols])
    df = pd.get_dummies(df, columns=cat_cols)
    return df

def format_model_name(filename):
    return filename.replace('.pkl', '').replace('_', ' ').title()

# ----------- Page Layout -----------

styled_header("🚀 Upload & Predict")

uploaded_file = st.file_uploader("📄 Upload CSV for Churn Prediction", type=["csv"])
predict_button = st.button("🔮 Predict")

model_dir = r"C:\\Users\\vijaykumark\\Desktop\\DS_PROJECT\\churn_app\\models"
scaler_path = os.path.join(model_dir, "scaler.pkl")

if uploaded_file and predict_button:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("📄 Uploaded Data Preview")
        st.dataframe(df.head())

        if not os.path.exists(scaler_path):
            st.error("❌ scaler.pkl file not found in model directory.")
            st.stop()

        scaler = joblib.load(scaler_path)

        original_cat_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                             'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                             'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod']
        original_num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
        df["SeniorCitizen"] = df["SeniorCitizen"].astype("O")
        df.dropna(inplace=True)

        target_cols = [col for col in ["Churn", "churned"] if col in df.columns]
        X_input = df.drop(columns=target_cols)

        df_processed = preprocess_input(X_input, original_cat_cols, original_num_cols, scaler)

        train_dummy_cols = joblib.load(os.path.join(model_dir, "train_dummy_columns.pkl"))
        df_processed = df_processed.reindex(columns=train_dummy_cols, fill_value=0)

        # ---------------- Tabs ----------------
        summary_tab, prediction_tab = st.tabs(["📄 Summary", "🧠 All Predictions"])

        with summary_tab:
            st.write("💾 **Data Overview (First 5 Rows)**")
            st.dataframe(df.head(5))

            st.write("📊 **Statistical Summary (All Columns)**")
            st.dataframe(df.describe(include='all'))

            st.write("📄 **Null Values in Data**")
            st.dataframe(df.isnull().sum().reset_index().rename(columns={'index': 'Column', 0: 'Null Count'}))

            st.write("📃 **Column Data Types**")
            st.dataframe(pd.DataFrame(df.dtypes, columns=["Data Type"]).reset_index().rename(columns={"index": "Column"}))

        with prediction_tab:
            model_preds = {}
            model_scores = []

            exclude_files = {"scaler.pkl", "train_dummy_columns.pkl", "num_cols.pkl"}
            model_files = [f for f in os.listdir(model_dir) if f.endswith(".pkl") and f not in exclude_files]

            prediction_tabs = st.tabs([format_model_name(f) for f in model_files])

            best_model_name = None
            best_accuracy = -1
            best_prediction_df = None

            for tab, file in zip(prediction_tabs, model_files):
                with tab:
                    model_path = os.path.join(model_dir, file)
                    try:
                        model = joblib.load(model_path)
                        preds = model.predict(df_processed)
                        proba = model.predict_proba(df_processed)[:, 1] if hasattr(model, "predict_proba") else [np.nan] * len(preds)

                        df_result = df.copy()
                        df_result["Churn Prediction"] = preds
                        df_result["Churn Probability"] = np.round(proba, 4)
                        model_preds[file] = df_result

                        st.write(f"📊 Predictions by **{format_model_name(file)}**")
                        st.dataframe(df_result)

                        if "Churn" in df.columns:
                            acc = accuracy_score(df["Churn"], preds)
                            try:
                                fpr, tpr, _ = roc_curve(df["Churn"], proba)
                                auc_score = roc_auc_score(df["Churn"], proba)

                                st.success(f"✅ Accuracy: {acc:.4f}")
                                st.info(f"📈 AUC-ROC Score: {auc_score:.4f}")

                                fig = go.Figure()
                                fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC = {auc_score:.2f})',
                                                         line=dict(color='blue', width=2)))
                                fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Guess',
                                                         line=dict(color='gray', dash='dash')))
                                fig.update_layout(title="Receiver Operating Characteristic (ROC Curve)",
                                                  xaxis_title="False Positive Rate",
                                                  yaxis_title="True Positive Rate",
                                                  width=700, height=500, template='plotly_white')

                                st.plotly_chart(fig, use_container_width=True)

                            except Exception as e:
                                st.warning(f"⚠️ Could not compute or plot AUC-ROC: {e}")

                            model_scores.append((file, acc))

                        else:
                            if best_prediction_df is None:
                                best_model_name = file
                                best_prediction_df = df_result

                        st.download_button(label="⬇️ Download Predictions CSV",
                                           data=df_result.to_csv(index=False),
                                           file_name=f"{file.replace('.pkl', '')}_predictions.csv",
                                           mime="text/csv")
                    except Exception as e:
                        st.warning(f"⚠️ Error loading/predicting with model {file}: {e}")

    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
