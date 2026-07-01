import pandas as pd, joblib, streamlit as st

st.title('Loan Default Predictor')
model = joblib.load('loan_default_model.pkl')
sample = pd.read_csv('sample_custom_input.csv').head(0)

inputs = {}
for col in sample.columns:
    if col in ['loan_id','customer_id','disbursal_date']:
        continue
    if col in ['gender','education','marital_status','employment_type','loan_purpose','loan_product','repayment_status','loan_source_type','state','district','city','urban_rural_flag','branch_code','branch_region','population_density_band','economic_activity_type','channel','has_mortgage','has_dependents','has_cosigner']:
        inputs[col] = st.text_input(col, '')
    else:
        inputs[col] = st.number_input(col, value=0.0)

if st.button('Predict'):
    row = pd.DataFrame([inputs])
    pred = model.predict(row)[0]
    prob = model.predict_proba(row)[0,1]
    st.success(f'Default Prediction: {int(pred)}')
    st.write(f'Default Probability: {prob:.4f}')
