# 🏥 Medical Insurance Cost Predictor

A complete end-to-end Machine Learning project that predicts annual medical
insurance costs based on personal health and demographic information.
Built with Python, trained on real-world data, deployed via Streamlit and Docker.

---

## 🎯 Problem Statement

Predict the annual medical insurance cost of a person based on their
health and demographic features such as age, BMI, smoking status, and region.

---

## 🔗 Links

| Resource | Link |
|---|---|
| 🌐 Live Demo | https://medicalinsurancecost-c8.streamlit.app/ |
| 🐳 Docker Hub | https://hub.docker.com/r/sagittarius1377/medical-insurance |
| 📓 Notebook | ./src.ipynb |

---

## 📌 Project Overview

| Item | Detail |
|---|---|
| Problem Type | Supervised Learning — Regression |
| Dataset | Medical Cost Personal Dataset |
| Total Records | 1338 patients |
| Features | 6 (age, sex, bmi, children, smoker, region) |
| Target | Annual insurance charges ($) |
| Best Model | XGBoost (after hyperparameter tuning) |
| R² Score | 0.8798 |
| MAE | $2,545.93 |
| RMSE | $4,319.03 |

---

## 📁 Project Structure

```
medical_insurance_cost/
├── Dockerfile
├── .dockerignore
└── medical_insurance/
    ├── src.ipynb
    ├── train_model.py
    ├── app.py
    ├── requirements.txt
    ├── insurance.csv
    ├── model.pkl
    ├── scaler.pkl
    ├── importance.csv
    └── model_stats.pkl
```

---

## 📊 Dataset

| Feature | Type | Description |
|---|---|---|
| age | Numeric | Age (18–64) |
| sex | Categorical | male / female |
| bmi | Numeric | Body Mass Index |
| children | Numeric | Number of dependents |
| smoker | Categorical | yes / no |
| region | Categorical | 4 regions |
| charges | Numeric | Target |

---

## 🔍 Key Insights

- Smokers pay **3.8× more**
- Smoker + Obese = **highest cost group**
- Age increases cost steadily
- Region has minimal impact

---

## 🤖 Models Compared

| Model | Tuned R² |
|---|---|
| Linear Regression | 0.7331 |
| Decision Tree | 0.8531 |
| Random Forest | 0.8734 |
| **XGBoost** | **0.8798 ✅** |

---

## ⚙️ Best Parameters (XGBoost)

- n_estimators = 500  
- max_depth = 3  
- learning_rate = 0.01  
- subsample = 0.8  
- colsample_bytree = 0.8  

---

## ⚙️ How It Works

User inputs are processed → scaled → passed to trained XGBoost model →  
prediction returned instantly via Streamlit app.

---

## 📦 Saved Artifacts

- model.pkl → trained XGBoost model  
- scaler.pkl → ensures same preprocessing during prediction  
- model_stats.pkl → model metrics and configuration  
- importance.csv → feature importance for visualization  

---

## 🖥️ App Features

- User input form  
- Prediction result  
- Risk assessment  
- Feature importance chart  
- Model info sidebar  

---

## 🚀 Run Locally

```bash
git clone https://github.com/Sagittarius1377/Medical_Insurance_Cost.git
cd Medical_Insurance_Cost
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

---

## 🐳 Docker

```bash
docker pull sagittarius1377/medical-insurance
docker run -p 8501:8501 sagittarius1377/medical-insurance
```

---

## 🏁 Conclusion

XGBoost achieved the best performance after tuning, outperforming
Random Forest and other models with an R² score of 0.8798.

---

## 🛠️ Tech Stack

Python, Pandas, NumPy, Scikit-learn, XGBoost, Streamlit, Docker

---

## 👩 Author

Bhumika  
GitHub: https://github.com/Sagittarius1377