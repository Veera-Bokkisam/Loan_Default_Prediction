import streamlit as st
import joblib
import pandas as pd

# ----------------------------
# LOAD MODEL + THRESHOLD
# ----------------------------
model = joblib.load("../models/loan_default_xgb_pipeline.pkl")

with open("../models/best_threshold.txt", "r") as f:
    BEST_THRESHOLD = float(f.read().strip())


# ----------------------------
# PREDICTION FUNCTION
# ----------------------------
def loan_default_predict(input_data):
    """
    input_data: dict of customer features
    """

    df = pd.DataFrame([input_data])

    # Predict probability of default
    prob = model.predict_proba(df)[:, 1][0]

    # Final decision using threshold
    pred = int(prob >= BEST_THRESHOLD)

    risk_label = "High Risk (Likely Default)" if pred == 1 else "Low Risk (Safe Customer)"

    return prob, pred, risk_label


# ----------------------------
# STREAMLIT UI
# ----------------------------
st.set_page_config(page_title="Loan Default Prediction", page_icon="💰", layout="wide")

st.title("💰 Loan Default Prediction App")
st.write("Enter borrower details below to predict loan default risk.")


# ----------------------------
# SIDEBAR INPUT FORM
# ----------------------------
st.sidebar.header("📝 Customer Input Details")

# MAIN INPUTS (only important features)
disbursed_amount = st.sidebar.number_input("Disbursed Amount (₹)", min_value=1000, max_value=5000000, value=100000)
asset_cost = st.sidebar.number_input("Asset Cost (₹)", min_value=10000, max_value=10000000, value=200000)
ltv = st.sidebar.number_input("LTV (Loan-to-Value Ratio %)", min_value=10.0, max_value=120.0, value=75.0)

Age = st.sidebar.number_input("Customer Age", min_value=18, max_value=80, value=35)
Loan_Age_Days = st.sidebar.number_input("Loan Age (Days)", min_value=0, max_value=5000, value=100)

Employment_Type = st.sidebar.selectbox("Employment Type", ["Salaried", "Self-employed"])

PERFORM_CNS_SCORE = st.sidebar.number_input("Credit Bureau Score", min_value=0, max_value=900, value=650)

NO_OF_INQUIRIES = st.sidebar.number_input("No. of Enquiries (last 6 months)", min_value=0, max_value=50, value=2)
DELINQ_6_MONTHS = st.sidebar.number_input("Delinquent Accts (last 6 months)", min_value=0, max_value=20, value=0)

PRI_NO_ACCTS = st.sidebar.number_input("PRI Number of Accounts", min_value=0, max_value=50, value=2)
PRI_ACTIVE_ACCTS = st.sidebar.number_input("PRI Active Accounts", min_value=0, max_value=50, value=1)
PRI_OVERDUE_ACCTS = st.sidebar.number_input("PRI Overdue Accounts", min_value=0, max_value=20, value=0)
PRI_SANCTIONED_AMOUNT = st.sidebar.number_input("PRI Sanctioned Amount", min_value=0, max_value=10000000, value=50000)
PRI_DISBURSED_AMOUNT = st.sidebar.number_input("PRI Disbursed Amount", min_value=0, max_value=10000000, value=50000)

CREDIT_HISTORY_LENGTH = st.sidebar.number_input("Credit History Length (Months)", min_value=0, max_value=500, value=60)
AVERAGE_ACCT_AGE = st.sidebar.number_input("Average Account Age (Months)", min_value=0, max_value=500, value=40)

State_ID = st.sidebar.number_input("State ID", min_value=1, max_value=50, value=10)
Current_pincode_ID = st.sidebar.number_input("Current Pincode ID", min_value=100000, max_value=999999, value=560001)


# ----------------------------
# MAKE INPUT DICTIONARY
# ----------------------------
input_data = {
    "disbursed_amount": disbursed_amount,
    "asset_cost": asset_cost,
    "ltv": ltv,
    "Age": Age,
    "Loan_Age_Days": Loan_Age_Days,
    "Employment.Type": Employment_Type,
    "PERFORM_CNS.SCORE": PERFORM_CNS_SCORE,
    "NO.OF_INQUIRIES": NO_OF_INQUIRIES,
    "DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS": DELINQ_6_MONTHS,
    "PRI.NO.OF.ACCTS": PRI_NO_ACCTS,
    "PRI.ACTIVE.ACCTS": PRI_ACTIVE_ACCTS,
    "PRI.OVERDUE.ACCTS": PRI_OVERDUE_ACCTS,
    "PRI.SANCTIONED.AMOUNT": PRI_SANCTIONED_AMOUNT,
    "PRI.DISBURSED.AMOUNT": PRI_DISBURSED_AMOUNT,
    "CREDIT.HISTORY.LENGTH_MONTHS": CREDIT_HISTORY_LENGTH,
    "AVERAGE.ACCT.AGE_MONTHS": AVERAGE_ACCT_AGE,
    "State_ID": State_ID,
    "Current_pincode_ID": Current_pincode_ID
}


# ----------------------------	
# RUN PREDICTION
# ----------------------------
if st.button("🔍 Predict Default Risk"):
    prob, pred, risk_label = loan_default_predict(input_data)

    st.subheader("📊 Prediction Result")
    st.write(f"**Probability of Default:** `{prob:.4f}`")
    st.write(f"**Threshold Used:** `{BEST_THRESHOLD}`")

    if pred == 1:
        st.error("🔴 **HIGH RISK – Likely Default**")
    else:
        st.success("🟢 **LOW RISK – Safe Customer**")

    st.info(f"**Risk Category:** {risk_label}")