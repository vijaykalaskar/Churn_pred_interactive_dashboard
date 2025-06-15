import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_validate, RandomizedSearchCV
import plotly.graph_objects as go

# ------------------ UI Functions ------------------
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

# ------------------ Page Header ------------------
styled_header("🛠️ Model Development")
fade_in_markdown("Train and evaluate baseline machine learning models using cross-validation.")

# ------------------ Load Data ------------------
df = pd.read_csv("data/telco_train.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df["SeniorCitizen"] = df["SeniorCitizen"].astype("O")
df.dropna(inplace=True)
df["Churn"] = df["churned"].map({1: 1, 0: 0})
cat_cols = df.select_dtypes(include="object").columns.tolist()
num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = [col for col in cat_cols if col not in ["Churn", "customerID", "churned"]]

# ------------------ Encoding & Scaling ------------------
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    return pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)

dff = one_hot_encoder(df.copy(), cat_cols)

# Re-identify numeric columns after encoding
num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
num_cols = [col for col in num_cols if col not in ["Churn", "churned"]]

# Scale numerical columns
scaler = RobustScaler()
dff[num_cols] = scaler.fit_transform(dff[num_cols])

# Save the scaler and numerical columns
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(num_cols, "models/num_cols.pkl")

# Define target and drop unused columns
y = dff["Churn"]
drop_cols = [col for col in dff.columns if col.lower() in ['churn', 'churned', 'customerid'] or col.lower().startswith('churn_')]
X = dff.drop(columns=drop_cols)

# Save dummy column structure
train_dummy_cols = X.columns.tolist()
joblib.dump(train_dummy_cols, "models/train_dummy_columns.pkl")

# ------------------ Model Definitions ------------------
models = [
    ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=12345)),
    ("K-Nearest Neighbors", KNeighborsClassifier()),
    ("Decision Tree (CART)", DecisionTreeClassifier(random_state=12345)),
    ("Random Forest", RandomForestClassifier(random_state=12345)),
    ("XGBoost", XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=12345)),
    ("LightGBM", LGBMClassifier(random_state=12345)),
    ("CatBoost", CatBoostClassifier(verbose=False, random_state=12345)),
    ("Gaussian Naive Bayes", GaussianNB()),
    ("Support Vector Machine", SVC(probability=True, random_state=12345)),
    ("Gradient Boosting", GradientBoostingClassifier(random_state=12345)),
    ("Extra Trees", ExtraTreesClassifier(random_state=12345))
]

# ------------------ Cached Evaluation ------------------
@st.cache_data
def evaluate_models(X, y):
    metrics_list = []
    for name, model in models:
        cv_results = cross_validate(model, X, y, cv=5,
                                    scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])

        metrics_list.append({
            "Model": name,
            "Accuracy": round(cv_results['test_accuracy'].mean(), 4),
            "AUC": round(cv_results['test_roc_auc'].mean(), 4),
            "Recall": round(cv_results['test_recall'].mean(), 4),
            "Precision": round(cv_results['test_precision'].mean(), 4),
            "F1": round(cv_results['test_f1'].mean(), 4)
        })
    return pd.DataFrame(metrics_list).sort_values(by="Accuracy", ascending=False)

# ------------------ Hyperparameter Tuning ------------------
@st.cache_data
def tune_models(X, y):
    model_save_path = r"C:\\Users\\vijaykumark\\Desktop\\DS_PROJECT\\churn_app\\models"
    os.makedirs(model_save_path, exist_ok=True)

    tuned_results = []
    best_accuracy = 0
    best_model_object = None

    param_grid = {
        "Logistic Regression": {"C": [0.01, 0.1, 1, 10], "solver": ["lbfgs", "liblinear"]},
        "K-Nearest Neighbors": {"n_neighbors": list(range(3, 11)), "weights": ["uniform", "distance"]},
        "Decision Tree (CART)": {"max_depth": [3, 5, 10, None], "min_samples_split": [2, 5, 10]},
        "Random Forest": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20], "min_samples_split": [2, 5]},
        "XGBoost": {"n_estimators": [50, 100], "max_depth": [3, 5, 7], "learning_rate": [0.01, 0.1, 0.2]},
        "LightGBM": {"n_estimators": [50, 100], "num_leaves": [15, 31, 63], "learning_rate": [0.01, 0.1, 0.2]},
        "CatBoost": {"iterations": [100, 200], "learning_rate": [0.01, 0.1], "depth": [4, 6, 8]},
        "Gaussian Naive Bayes": {},
        "Support Vector Machine": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
        "Gradient Boosting": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1], "max_depth": [3, 5]},
        "Extra Trees": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]}
    }

    for model_name, model in models:
        grid = param_grid.get(model_name, {})
        rs = RandomizedSearchCV(model, grid, n_iter=10, cv=5, n_jobs=-1, scoring='accuracy', random_state=123)
        rs.fit(X, y)
        acc = rs.best_score_

        tuned_results.append({
            "Model": model_name,
            "Best Params": rs.best_params_,
            "Tuned Accuracy": round(acc, 4)
        })

        model_filename = f"{model_name.replace(' ', '_').replace('(', '').replace(')', '').lower()}.pkl"
        joblib.dump(rs.best_estimator_, os.path.join(model_save_path, model_filename))

        if acc > best_accuracy:
            best_accuracy = acc
            best_model_object = {
                "name": model_name,
                "accuracy": acc,
                "params": rs.best_params_,
                "object": rs.best_estimator_
            }

    return pd.DataFrame(tuned_results).sort_values(by="Tuned Accuracy", ascending=False), best_model_object

# ------------------ Run Evaluation ------------------
results_df = evaluate_models(X, y)
tuned_df, best_model = tune_models(X, y)

# ------------------ Tabs ------------------
tabs = st.tabs([
    "📘 Metric Meaning",
    "📊 Evaluation Chart",
    "📊 Model Performance on Cross-Validation Graphs",
    "⚙️ Hyperparameter Tuning",
    "🏆 Best Model Summary"
])

# Tab 1: Metric Meaning
with tabs[0]:
    st.markdown("""
    ### 📘 Evaluation Metric Definitions
    - **Accuracy**: Proportion of total correct predictions.
    - **AUC (ROC AUC)**: Area under the ROC curve; shows the ability of the model to distinguish between classes.
    - **Recall**: True positive rate - how many actual positives were correctly predicted.
    - **Precision**: How many predicted positives were correct.
    - **F1 Score**: Harmonic mean of Precision and Recall.
    """)

# Tab 2: Evaluation Chart
with tabs[1]:
    st.subheader("📋 Model Evaluation Metrics")
    st.dataframe(results_df, use_container_width=True)
    st.download_button("📥 Download CSV", data=results_df.to_csv(index=False), file_name="model_results.csv", mime="text/csv")

# Tab 3: Cross-Validation Graphs
with tabs[2]:
    st.subheader("📊 Model Performance on Cross-Validation Graphs")
    fig = go.Figure()
    colors = ["purple", "green", "blue", "orange", "red"]

    for metric, color in zip(["Accuracy", "AUC", "Recall", "Precision", "F1"], colors):
        fig.add_trace(go.Bar(
            x=results_df["Model"],
            y=results_df[metric],
            name=metric,
            marker_color=color,
            text=results_df[metric],
            textposition='auto'
        ))

    fig.update_layout(
        barmode='group',
        title="🔍 Model Comparison (Cross-Validated)",
        xaxis_title="Model",
        yaxis_title="Metric Score",
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)

# Tab 4: Hyperparameter Tuning
with tabs[3]:
    st.subheader("⚙️ Hyperparameter Tuning Results")
    st.dataframe(tuned_df, use_container_width=True)
    st.download_button("📥 Download Tuned Results", data=tuned_df.to_csv(index=False), file_name="tuned_results.csv", mime="text/csv")

# Tab 5: Best Model Summary
with tabs[4]:
    st.subheader("🏆 Best Performing Model After Tuning")
    st.markdown(f"""
    **Model**: {best_model['name']}  
    **Accuracy**: {round(best_model['accuracy'], 4)}  
    **Best Hyperparameters**: {best_model['params']}  

    ✅ Based on tuning results, **{best_model['name']}** is the best model. The tuned model has been saved to disk as `best_model_{best_model['name']}.pkl`.
    """)

    # Save the final best model separately
    joblib.dump(best_model['object'], f"models/best_model_{best_model['name']}.pkl")
