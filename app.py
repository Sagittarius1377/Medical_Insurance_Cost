# ============================================
# app.py  — Medical Insurance Cost Predictor
# Run with: streamlit run app.py
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# ---- Page config (must be first streamlit command) ----
st.set_page_config(
    page_title = "Insurance Cost Predictor",
    page_icon  = "🏥",
    layout     = "wide"
)

# ---- Load all saved files ----
@st.cache_resource
def load_files():
    with open("model.pkl",       "rb") as f: model      = pickle.load(f)
    with open("scaler.pkl",      "rb") as f: scaler     = pickle.load(f)
    with open("model_stats.pkl", "rb") as f: stats      = pickle.load(f)
    importance = pd.read_csv("importance.csv")
    return model, scaler, importance, stats

model, scaler, importance_df, stats = load_files()

# ================================================
# SIDEBAR — Model Info
# ================================================

st.sidebar.title("📊 Model Information")
st.sidebar.divider()

st.sidebar.metric("Model Used",  stats['model_name'])
st.sidebar.metric("R² Score",    stats['r2'])
st.sidebar.metric("Avg Error",  f"${stats['mae']:,}")
st.sidebar.metric("Trained On", f"{stats['train_rows']} patients")

st.sidebar.divider()
st.sidebar.subheader("⚙️ Best Tuned Settings")
for param, value in stats['best_params'].items():
    st.sidebar.write(f"**{param}** → {value}")

st.sidebar.divider()
st.sidebar.subheader("📌 Dataset Insights")
st.sidebar.write("- Smokers pay **3.8× more** on average")
st.sidebar.write("- Smoker + Obese = highest risk group")
st.sidebar.write("- Age is 2nd strongest cost driver")
st.sidebar.write("- Region has almost no effect")

# ================================================
# MAIN PAGE — Title
# ================================================

st.title("🏥 Medical Insurance Cost Predictor")
st.write("Enter your details to get an estimated annual insurance cost.")
st.divider()

# ================================================
# INPUT SECTION
# ================================================

st.subheader("👤 Enter Your Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider(
        "Age",
        min_value = 18,
        max_value = 64,
        value     = 30
    )

    sex = st.selectbox("Sex", options=["male", "female"])

    bmi = st.number_input(
        "BMI",
        min_value = 10.0,
        max_value = 60.0,
        value     = 25.0,
        step      = 0.1,
        help      = "Normal BMI range is 18.5 – 24.9. Cannot be below 10 or above 60."
    )

with col2:
    children = st.slider(
        "Number of Children",
        min_value = 0,
        max_value = 5,
        value     = 0
    )

    smoker = st.selectbox("Smoker?", options=["no", "yes"])

    region = st.selectbox(
        "Region",
        options = ["northeast", "northwest", "southeast", "southwest"]
    )

st.divider()

# ================================================
# PREDICT BUTTON
# ================================================

predict_btn = st.button("🔮 Predict My Insurance Cost",
                         use_container_width=True)

if predict_btn:

    # ================================================
    # LAYER 1 — INPUT VALIDATION
    # ================================================
    # WHY: Collect ALL errors first, show them together
    #      instead of one at a time

    errors = []

    if bmi < 10.0:
        errors.append("❌ BMI cannot be less than 10. Please enter a valid BMI.")

    if bmi > 60.0:
        errors.append("❌ BMI cannot be more than 60. Please enter a valid BMI.")

    if age < 18:
        errors.append("❌ Age must be at least 18.")

    if age > 64:
        errors.append("❌ Age cannot be more than 64.")

    if children < 0:
        errors.append("❌ Number of children cannot be negative.")

    # ---- If any errors exist → show and STOP ----
    if errors:
        for error in errors:
            st.error(error)
        st.stop()
        # st.stop() → nothing below runs if validation fails

    # ================================================
    # LAYER 2 — BUILD INPUT & PREDICT
    # ================================================

    input_data = pd.DataFrame({
        'age'              : [age],
        'bmi'              : [bmi],
        'children'         : [children],
        'sex_male'         : [1 if sex    == 'male'      else 0],
        'smoker_yes'       : [1 if smoker == 'yes'       else 0],
        'region_northwest' : [1 if region == 'northwest' else 0],
        'region_southeast' : [1 if region == 'southeast' else 0],
        'region_southwest' : [1 if region == 'southwest' else 0],
    })

    input_scaled = scaler.transform(input_data)
    prediction   = model.predict(input_scaled)[0]

    # ================================================
    # RESULT SECTION
    # ================================================

    st.subheader("💰 Prediction Result")

    res_col1, res_col2, res_col3 = st.columns(3)

    with res_col1:
        st.metric(
            label = "Your Estimated Annual Cost",
            value = f"${prediction:,.2f}"
        )
    with res_col2:
        st.metric(
            label = "Dataset Average Cost",
            value = f"${stats['avg_charges']:,}"
        )
    with res_col3:
        diff = prediction - stats['avg_charges']
        st.metric(
            label = "Difference from Average",
            value = f"${abs(diff):,.2f}",
            delta = f"{'above' if diff > 0 else 'below'} average"
        )

    st.divider()

    # ================================================
    # RISK ASSESSMENT
    # ================================================

    st.subheader("🎯 Risk Assessment")

    if prediction < 10000:
        st.success("🟢 LOW RISK — Your estimated cost is well below average")
    elif prediction < 20000:
        st.warning("🟡 MEDIUM RISK — Your cost is above the dataset average")
    else:
        st.error("🔴 HIGH RISK — Your cost is significantly above average")

    # specific warnings based on inputs
    if smoker == "yes":
        st.warning("⚠️ Smoking is the #1 driver of insurance cost in this model.")

    if bmi >= 30 and smoker == "yes":
        st.error("🚨 Smoker + BMI over 30 is the most expensive combination — avg $41,557 in our dataset.")

    if age >= 50 and smoker == "yes":
        st.error("🚨 Being over 50 and a smoker puts you in the highest cost group.")

    st.divider()

    # ================================================
    # FEATURE IMPORTANCE CHART
    # ================================================

    st.subheader("📊 What Drives Insurance Cost?")
    st.write("This shows which features our XGBoost model relies on most:")

    fig, ax = plt.subplots(figsize=(8, 4))

    colors = ['tomato' if f == 'smoker_yes' else 'steelblue'
              for f in importance_df['Feature']]

    ax.barh(importance_df['Feature'],
            importance_df['Importance'],
            color  = colors,
            edgecolor = 'white')
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importance — XGBoost Model")
    plt.tight_layout()
    st.pyplot(fig)

    st.caption("🔴 Red bar = smoker_yes — the single most powerful predictor")

    st.divider()

    # ================================================
    # INPUT SUMMARY
    # ================================================

    st.subheader("📋 Your Input Summary")

    summary = pd.DataFrame({
        'Feature' : ['Age', 'Sex', 'BMI', 'Children', 'Smoker', 'Region'],
        'Value'   : [age, sex, bmi, children, smoker, region]
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.divider()

    # ================================================
    # KEY INSIGHTS
    # ================================================

    st.subheader("🔍 Key Insights from the Data")

    ins_col1, ins_col2 = st.columns(2)

    with ins_col1:
        st.info("🚬 Smokers pay **$32,050** avg vs **$8,434** for non-smokers")
        st.info("📈 Each decade of age adds ~$2,500 to annual cost")

    with ins_col2:
        st.info("⚖️ Smoker + Obese combination averages **$41,557/year**")
        st.info("🗺️ Region has almost no effect on cost (correlation: 0.07)")