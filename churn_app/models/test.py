import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler

# ---------- Paths ----------
model_path = "C:/Users/vijaykumark/Desktop/DS_PROJECT/churn_app/models/catboost.pkl"
scaler_path = "C:/Users/vijaykumark/Desktop/DS_PROJECT/churn_app/models/scaler1.pkl"
columns_path = "C:/Users/vijaykumark/Desktop/DS_PROJECT/churn_app/models/train_dummy_columns1.pkl"
test_file = "C:/Users/vijaykumark/Desktop/DS_PROJECT/churn_app/telco_test.csv"

# ---------- Load Model and Artifacts ----------
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
train_cols = joblib.load(columns_path)

# ---------- Preprocessing Utilities ----------
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    return pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)

# ---------- Load and Prepare Test Data ----------
df = pd.read_csv(test_file)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["SeniorCitizen"] = df["SeniorCitizen"].astype("O")
df.dropna(inplace=True)

# Define categorical and numerical columns
cat_cols = df.select_dtypes(include="object").columns.tolist()
num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = [col for col in cat_cols if col.lower() not in ["churn", "churned", "customerid"]]

# Apply one-hot encoding
df_encoded = one_hot_encoder(df.copy(), cat_cols)

# Drop churn/customer ID columns if present
drop_cols = [col for col in df_encoded.columns if col.lower() in ["churn", "churned", "customerid"] or col.lower().startswith("churn_")]
df_encoded.drop(columns=drop_cols, inplace=True, errors="ignore")

# Add missing columns and fix order
for col in train_cols:
    if col not in df_encoded.columns:
        df_encoded[col] = 0

df_encoded = df_encoded[train_cols]  # Align column order

# Scale all numeric columns at once (must be same as during training)
df_encoded[num_cols] = scaler.transform(df_encoded[num_cols])


# ---------- Make Prediction ----------
predictions = model.predict(df_encoded)
probas = model.predict_proba(df_encoded)[:, 1]  # probability of churn

# ---------- Output ----------
df["Prediction"] = predictions
df["Churn_Probability"] = probas

print(df[["customerID", "Prediction", "Churn_Probability"]].head())
