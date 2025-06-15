# Stop the Churn – Customer Churn Prediction Dashboard

A business-critical Streamlit dashboard for analyzing, visualizing, and predicting customer churn using machine learning models.

## Project Structure

```
Home.py                  # Main Streamlit app (dashboard entry point)
.streamlit/              # Streamlit configuration files
assets/                  # Static assets (images, icons, etc.)
catboost_info/           # CatBoost model training logs and info
data/
  telco_train.csv        # Training data
  telco_test.csv         # Test data
Document/
  Churn_pred_doc.docx    # Project documentation (Word)
  Churn_pred_doc.pdf     # Project documentation (PDF)
models/                  # Saved ML models and preprocessing objects
pages/                   # Streamlit multipage app scripts
testing_data/            # Additional test datasets
utils/                   # Utility scripts and helpers
```

## Features

- 📊 **Data Visualization:** Explore churn distribution, trends, and correlations.
- 🛠️ **Model Development:** Train, tune, and compare multiple ML models (CatBoost, LightGBM, Random Forest, etc.).
- 🚀 **Upload & Predict:** Upload your own CSV data and get churn predictions from trained models.
- 📄 **Documentation:** Detailed project report in [Document/](Document/).

## Getting Started

1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

2. **Run the app:**
   ```sh
   streamlit run Home.py
   ```

3. **Navigate:** Use the sidebar to explore data visualization, model development, and prediction pages.

## Data

- Located in [data/](data/) folder.
- Example: `telco_train.csv`, `telco_test.csv`

## Models

- Pretrained models in [models/](models/): CatBoost, LightGBM, Random Forest, Logistic Regression, etc.

## Documentation

- See [Document/Churn_pred_doc.pdf](Document/Churn_pred_doc.pdf) for detailed methodology and results.

---

**Author:** Vijaykumar Kalaskar  
