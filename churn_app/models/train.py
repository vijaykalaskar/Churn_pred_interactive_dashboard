import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier

# ---------------- Load and Prepare Data ----------------
df = pd.read_csv(r"C:\Users\vijaykumark\Desktop\DS_PROJECT\churn_app\data\telco_train.csv")

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df["SeniorCitizen"] = df["SeniorCitizen"].astype("O")
df.dropna(inplace=True)
df["Churn"] = df["churned"].map({1: 1, 0: 0})

cat_cols = df.select_dtypes(include="object").columns.tolist()
cat_cols = [col for col in cat_cols if col not in ["Churn", "customerID", "churned"]]

num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
num_cols = [col for col in num_cols if col not in ["Churn", "churned"]]

# ---------------- One-Hot Encoding ----------------
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    return pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)

df_encoded = one_hot_encoder(df.copy(), cat_cols)

# ---------------- Scaling ----------------
scaler = RobustScaler()
df_encoded[num_cols] = scaler.fit_transform(df_encoded[num_cols])

# ---------------- Save Scaler and Columns ----------------
joblib.dump(scaler, "scaler1.pkl")
joblib.dump(num_cols, "num_cols1.pkl")

# ---------------- Feature Engineering ----------------
y = df_encoded["Churn"]
drop_cols = [col for col in df_encoded.columns if col.lower() in ["churn", "churned", "customerid"] or col.lower().startswith("churn_")]
X = df_encoded.drop(columns=drop_cols)

train_dummy_cols = X.columns.tolist()
joblib.dump(train_dummy_cols, "train_dummy_columns1.pkl")

# ---------------- Train a Sample Model ----------------
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

joblib.dump(model, "rf_model1.pkl")
