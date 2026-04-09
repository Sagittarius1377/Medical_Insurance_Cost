# ============================================
# train_model.py
# Run this file ONCE to train and save model
# Command: python train_model.py
# ============================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import pickle

print("="*45)
print("   MEDICAL INSURANCE — MODEL TRAINING")
print("="*45)

# ---- Step 1: Load data ----
df = pd.read_csv("insurance.csv")
print(f"\n✅ Data loaded — {df.shape[0]} rows, {df.shape[1]} columns")

# ---- Step 2: Encode ----
df_encoded = pd.get_dummies(df, columns=['sex', 'smoker', 'region'],
                             drop_first=True, dtype=int)

X = df_encoded.drop(columns=['charges'])
y = df_encoded['charges']

# ---- Step 3: Split ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"✅ Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

# ---- Step 4: Scale ----
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print("✅ Scaling done")

# ---- Step 5: Train XGBoost with YOUR best params ----
print("\n🔍 Training XGBoost with best tuned settings...")

model = XGBRegressor(
    subsample        = 0.8,
    n_estimators     = 500,
    max_depth        = 3,
    learning_rate    = 0.01,
    colsample_bytree = 0.8,
    random_state     = 42,
    verbosity        = 0
)

model.fit(X_train_scaled, y_train)
print("✅ Model trained!")

# ---- Step 6: Verify scores ----
y_pred = model.predict(X_test_scaled)
mae    = mean_absolute_error(y_test, y_pred)
rmse   = mean_squared_error(y_test, y_pred) ** 0.5
r2     = r2_score(y_test, y_pred)

print(f"\n--- Final Model Scores ---")
print(f"MAE  : ${mae:,.2f}")
print(f"RMSE : ${rmse:,.2f}")
print(f"R²   : {r2:.4f}")

# ---- Step 7: Save feature importance ----
importance_df = pd.DataFrame({
    'Feature'   : X_train.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False).reset_index(drop=True)

# ---- Step 8: Save everything ----
# We save 4 things:
# model.pkl        → the trained XGBoost model
# scaler.pkl       → the fitted scaler (must match training)
# importance.pkl   → feature importances for the chart in app
# model_stats.pkl  → scores and info shown in the sidebar

model_stats = {
    'model_name'  : 'XGBoost',
    'r2'          : round(r2, 4),
    'mae'         : round(mae, 2),
    'rmse'        : round(rmse, 2),
    'train_rows'  : X_train.shape[0],
    'features'    : X_train.shape[1],
    'avg_charges' : round(df['charges'].mean(), 2),
    'best_params' : {
        'subsample'        : 0.8,
        'n_estimators'     : 500,
        'max_depth'        : 3,
        'learning_rate'    : 0.01,
        'colsample_bytree' : 0.8
    }
}

with open("model.pkl",      "wb") as f: pickle.dump(model,        f)
with open("scaler.pkl",     "wb") as f: pickle.dump(scaler,       f)
importance_df.to_csv("importance.csv", index=False)
print("✅ importance.csv saved!")

with open("model_stats.pkl","wb") as f: pickle.dump(model_stats,  f)

print("\n✅ model.pkl        saved!")
print("✅ scaler.pkl       saved!")
print("✅ importance.pkl   saved!")
print("✅ model_stats.pkl  saved!")
print("\n🚀 Now run: streamlit run app.py")
print("="*45)