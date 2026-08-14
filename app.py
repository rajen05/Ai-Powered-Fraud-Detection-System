"""
Streamlit app for the Credit Card Fraud Detection project.

Run with:  streamlit run app.py

Two tabs:
1. Predict — manually enter transaction values (or load a random sample
   row) and see the model's fraud probability, with an adjustable
   decision threshold so you can visualize the precision/recall tradeoff.
2. Model Insights — the EDA charts and model comparison table from the
   notebook, so the app doubles as a mini case study.
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Credit Card Fraud Detector", page_icon="💳", layout="wide")

MODEL_PATH = "models/fraud_model.pkl"
DATA_PATH = "data/creditcard.csv"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_sample_data():
    """Loads a small sample for the 'load random row' feature and the charts.
    Falls back gracefully if the full dataset isn't present."""
    try:
        df = pd.read_csv(DATA_PATH)
        return df
    except FileNotFoundError:
        return None


st.title("💳 Credit Card Fraud Detector")
st.caption(
    "Trained on the ULB Credit Card Fraud dataset (284,807 transactions, "
    "0.17% fraud). Built to demonstrate handling severe class imbalance — "
    "see the **Model Insights** tab for why accuracy alone is misleading here."
)

tab_predict, tab_insights = st.tabs(["🔍 Predict", "📊 Model Insights"])

# ----------------------------------------------------------------------
# TAB 1: PREDICT
# ----------------------------------------------------------------------
with tab_predict:
    try:
        model = load_model()
    except FileNotFoundError:
        st.error(
            "No trained model found at `models/fraud_model.pkl`. "
            "Run `python src/train.py` first (see STEPS.md)."
        )
        st.stop()

    df_sample = load_sample_data()

    st.subheader("Enter transaction details")

    col_a, col_b = st.columns([1, 2])
    with col_a:
        if df_sample is not None and st.button("🎲 Load a random transaction"):
            row = df_sample.sample(1).iloc[0]
            st.session_state["prefill"] = row.to_dict()

    prefill = st.session_state.get("prefill", {})

    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        with c1:
            amount = st.number_input(
                "Transaction Amount ($)", min_value=0.0,
                value=float(prefill.get("Amount", 50.0)), step=1.0
            )
        with c2:
            time_val = st.number_input(
                "Time (seconds since first transaction in dataset)",
                min_value=0.0, value=float(prefill.get("Time", 0.0)), step=1.0
            )

        st.markdown(
            "**V1–V28** are PCA-anonymized features from the original dataset "
            "(this is standard for real financial data — raw features are "
            "never shared for privacy reasons). Defaults are 0; load a random "
            "transaction above to see realistic values."
        )
        v_cols = st.columns(4)
        v_values = {}
        for i in range(1, 29):
            col = v_cols[(i - 1) % 4]
            v_values[f"V{i}"] = col.number_input(
                f"V{i}", value=float(prefill.get(f"V{i}", 0.0)),
                step=0.1, format="%.4f", key=f"v_{i}"
            )

        submitted = st.form_submit_button("Check for fraud")

    threshold = st.slider(
        "Decision threshold (probability above this = flagged as fraud)",
        min_value=0.0, max_value=1.0, value=0.5, step=0.01,
        help=(
            "Lower threshold = catches more fraud but more false alarms "
            "(higher recall, lower precision). Higher threshold = fewer false "
            "alarms but more fraud slips through (higher precision, lower recall)."
        ),
    )

    if submitted:
        input_df = pd.DataFrame([{
            "Time": time_val, "Amount": amount, **v_values
        }])
        # Reorder columns to match training data (Time, V1-V28, Amount)
        col_order = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
        input_df = input_df[col_order]

        proba = model.predict_proba(input_df)[0, 1]
        flagged = proba >= threshold

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Fraud Probability", f"{proba*100:.2f}%")
        with col2:
            if flagged:
                st.error("🚨 FLAGGED as potential fraud")
            else:
                st.success("✅ Not flagged")

# ----------------------------------------------------------------------
# TAB 2: MODEL INSIGHTS
# ----------------------------------------------------------------------
with tab_insights:
    st.subheader("Why accuracy is the wrong metric here")
    st.markdown(
        """
        Only **0.17%** of transactions in this dataset are fraud. A model that
        predicts *"not fraud"* every single time would score **99.83% accuracy**
        while catching zero fraud cases. That's why this project evaluates
        models on **Precision, Recall, F1, and PR-AUC** instead.
        """
    )

    if df_sample is not None:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Class distribution**")
            fig, ax = plt.subplots(figsize=(5, 4))
            df_sample["Class"].value_counts().plot(kind="bar", ax=ax)
            ax.set_xticklabels(["Legit", "Fraud"], rotation=0)
            st.pyplot(fig)
        with c2:
            st.markdown("**Amount distribution (log scale)**")
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.hist(np.log1p(df_sample["Amount"]), bins=40)
            ax.set_xlabel("log(1 + Amount)")
            st.pyplot(fig)
    else:
        st.info(
            "Place `creditcard.csv` in the `data/` folder to see live EDA charts here."
        )

    st.markdown(
        """
        **Full model comparison, precision-recall curves, and imbalance-handling
        approach (class weighting vs SMOTE) are in `notebooks/01_eda_and_modeling.ipynb`.**
        """
    )

    st.markdown(
        """
        ### Design decisions worth noting
        - SMOTE (synthetic oversampling) was applied to the **training set only**
          — applying it before the train/test split would leak synthetic patterns
          into evaluation and produce misleadingly high scores.
        - The threshold slider above lets you see the precision/recall tradeoff
          directly: in a real fraud system, missing fraud (false negative) is
          usually costlier than a false alarm (false positive), which argues for
          favoring recall — but that's a business decision, not a purely technical one.
        """
    )
