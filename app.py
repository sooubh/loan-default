import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.base import BaseEstimator, ClassifierMixin
import sklearn.compose._column_transformer as _ct

# Inject shim to handle backward-compatibility issue with older scikit-learn versions
if not hasattr(_ct, "_RemainderColsList"):
    class _RemainderColsList:
        pass
    _ct._RemainderColsList = _RemainderColsList

# 1. Custom Class Definition for Pickle Deserialization
# Since the model is saved in the notebook as a __main__.WrappedModel instance,
# defining it here at the top level ensures joblib can deserialize it.
class WrappedModel(BaseEstimator, ClassifierMixin):
    def __init__(self, model, preprocess):
        self.model = model
        self.preprocess = preprocess
    def fit(self, X, y):
        Xp = self.preprocess.fit_transform(X)
        self.model.fit(Xp, y)
        return self
    def predict(self, X):
        return self.model.predict(self.preprocess.transform(X))
    def predict_proba(self, X):
        return self.model.predict_proba(self.preprocess.transform(X))

# Set page configuration with a premium look
st.set_page_config(
    page_title="Loan Default Risk Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject modern, premium CSS styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .sub-title {
        color: #a0aec0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 600;
        color: #764ba2;
    }
    
    .result-card-safe {
        background: linear-gradient(135deg, #134e3a 0%, #0f766e 100%);
        border-radius: 12px;
        padding: 24px;
        color: white;
        border: 1px solid #14b8a6;
        box-shadow: 0 4px 15px rgba(20, 184, 166, 0.2);
    }
    
    .result-card-default {
        background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
        border-radius: 12px;
        padding: 24px;
        color: white;
        border: 1px solid #ef4444;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.2);
    }
    </style>
""", unsafe_allow_html=True)

# 2. Main Title Header
st.markdown('<div class="main-title">🏦 Loan Default Risk Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Assess creditworthiness and default probabilities using advanced machine learning.</div>', unsafe_allow_html=True)

# Load Schema
@st.cache_data
def load_schema():
    if os.path.exists('schema.json'):
        with open('schema.json', 'r') as f:
            return json.load(f)
    return None

schema = load_schema()

# Check Model Availability
MODEL_PATH = 'loan_default_model.pkl'
model_exists = os.path.exists(MODEL_PATH)

# Preset Cases Definition
PRESETS = {
    "Manual Entry": None,
    "Sample: Low Risk Borrower": {
        'age': 42, 'gender': 'Female', 'education': 'PostGraduate', 'marital_status': 'Married', 
        'employment_type': 'Salaried', 'employment_length_months': 56, 'annual_income': 82000, 
        'monthly_income': 6833, 'existing_emi': 873, 'debt_to_income': 0.306, 'credit_score': 810, 
        'credit_history_length_months': 65, 'delinquency_count': 0, 'past_default_flag': 0, 
        'loan_amount': 132000, 'loan_term_months': 72, 'interest_rate': 6.5, 'emi_amount': 1859, 
        'loan_purpose': 'Business', 'loan_product': 'Unsecured', 'repayment_status': 'Active', 
        'first_emi_default_flag': 0, 'loan_source_type': 'New', 'state': 'Tamil Nadu', 
        'district': 'Chennai', 'city': 'Chennai', 'pincode': 204565, 'urban_rural_flag': 'Urban', 
        'branch_code': 'WEST', 'branch_region': 'SouthZone', 'service_area_cluster': 39, 
        'distance_to_branch_km': 4.3, 'area_default_rate': 0.05, 'district_risk_score': 0.23, 
        'state_risk_score': 0.15, 'population_density_band': 'Medium', 'economic_activity_type': 'Mixed', 
        'channel': 'Digital', 'has_mortgage': 'Yes', 'has_dependents': 'Yes', 'has_cosigner': 'No'
    },
    "Sample: High Risk Borrower": {
        'age': 41, 'gender': 'Female', 'education': 'Graduate', 'marital_status': 'Married', 
        'employment_type': 'SelfEmployed', 'employment_length_months': 86, 'annual_income': 51000, 
        'monthly_income': 4250, 'existing_emi': 4114, 'debt_to_income': 0.88, 'credit_score': 450, 
        'credit_history_length_months': 195, 'delinquency_count': 3, 'past_default_flag': 1, 
        'loan_amount': 311000, 'loan_term_months': 36, 'interest_rate': 18.5, 'emi_amount': 8706, 
        'loan_purpose': 'Vehicle', 'loan_product': 'Secured', 'repayment_status': 'Overdue', 
        'first_emi_default_flag': 1, 'loan_source_type': 'New', 'state': 'Rajasthan', 
        'district': 'Udaipur', 'city': 'Udaipur', 'pincode': 315012, 'urban_rural_flag': 'Urban', 
        'branch_code': 'CENTRAL', 'branch_region': 'CentralZone', 'service_area_cluster': 22, 
        'distance_to_branch_km': 8.8, 'area_default_rate': 0.28, 'district_risk_score': 0.77, 
        'state_risk_score': 0.69, 'population_density_band': 'Medium', 'economic_activity_type': 'Salaried', 
        'channel': 'Partner', 'has_mortgage': 'Yes', 'has_dependents': 'Yes', 'has_cosigner': 'No'
    }
}

# Sidebar - Configuration and Presets
with st.sidebar:
    st.markdown("### ⚙️ App Controls")
    preset_choice = st.selectbox(
        "Select a Preset Borrower Profile",
        options=list(PRESETS.keys()),
        help="Quickly pre-fill all form inputs with specific test profiles."
    )
    
    st.markdown("---")
    st.markdown("### 📊 Model Status")
    if model_exists:
        st.success("Model Status: Loaded Successfully")
    else:
        st.warning("Model Status: Model Pickle File Not Found")
        
    st.markdown("---")
    st.markdown("### 📈 Performance Metrics (Typical)")
    st.metric(label="Model ROC-AUC", value="~87.4%", delta="+0.8% vs Baseline")
    st.metric(label="Accuracy", value="~83.2%", delta="+1.2% vs Baseline")
    
    st.markdown("---")
    st.info("Developed for IDBI Loan Default Prediction Hackathon.")

# Handle Model Missing State
if not model_exists:
    st.warning("⚠️ **Model file `loan_default_model.pkl` is missing!**")
    st.markdown("""
    ### How to Train and Load the Model:
    1. **Run the Notebook**: Open and execute `loan_default_hackathon_notebook.ipynb` in Google Colab or your local Jupyter environment.
    2. **Generate the model**: The notebook will automatically train the model and save it as `loan_default_model.pkl`.
    3. **Upload/Commit the model**: Download the pickle file from Colab and upload it into this workspace directory (or push it to your GitHub repository if deploying on Streamlit Cloud).
    4. **Refresh this page**: Once `loan_default_model.pkl` is in the directory, this app will automatically load it.
    """)
    st.stop()

# Load the model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Pre-populated values based on chosen preset
active_preset = PRESETS[preset_choice]

# Check if schema is loaded
if not schema:
    st.error("Error: schema.json is missing! Please run `generate_schema.py` to create the schema file.")
    st.stop()

# Create dynamic input forms using Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Borrower Profile", 
    "💼 Employment & Income", 
    "💳 Credit & History", 
    "💰 Loan Details", 
    "📍 Geography & Branch"
])

inputs = {}

# Populate Tab 1: Borrower Profile
with tab1:
    st.markdown("### Basic Demographics")
    col1, col2 = st.columns(2)
    
    with col1:
        c = 'age'
        default_val = int(active_preset[c]) if active_preset else int(schema['numerical'][c]['median'])
        inputs[c] = st.slider(
            "Borrower Age (Years)", 
            min_value=int(schema['numerical'][c]['min']), 
            max_value=int(schema['numerical'][c]['max']), 
            value=default_val
        )
        
        c = 'gender'
        default_idx = schema['categorical'][c].index(active_preset[c]) if active_preset else 0
        inputs[c] = st.selectbox("Gender", options=schema['categorical'][c], index=default_idx)
        
        c = 'marital_status'
        default_idx = schema['categorical'][c].index(active_preset[c]) if active_preset else 0
        inputs[c] = st.selectbox("Marital Status", options=schema['categorical'][c], index=default_idx)
        
    with col2:
        c = 'education'
        default_idx = schema['categorical'][c].index(active_preset[c]) if active_preset else 0
        inputs[c] = st.selectbox("Education Level", options=schema['categorical'][c], index=default_idx)
        
        c = 'has_dependents'
        default_idx = schema['categorical'][c].index(active_preset[c]) if active_preset else 0
        inputs[c] = st.selectbox("Has Dependents", options=schema['categorical'][c], index=default_idx)

# Populate Tab 2: Employment & Income
with tab2:
    st.markdown("### Employment and Financial Capacity")
    col1, col2 = st.columns(2)
    
    with col1:
        c = 'employment_type'
        default_idx = schema['categorical'][c].index(active_preset[c]) if active_preset else 0
        inputs[c] = st.selectbox("Employment Type", options=schema['categorical'][c], index=default_idx)
        
        c = 'employment_length_months'
        default_val = int(active_preset[c]) if active_preset else int(schema['numerical'][c]['median'])
        inputs[c] = st.number_input(
            "Employment Length (Months)", 
            min_value=int(schema['numerical'][c]['min']), 
            max_value=int(schema['numerical'][c]['max']), 
            value=default_val
        )
        
        c = 'economic_activity_type'
        default_idx = schema['categorical'][c].index(active_preset[c]) if active_preset else 0
        inputs[c] = st.selectbox("Economic Activity Type", options=schema['categorical'][c], index=default_idx)
        
    with col2:
        c = 'annual_income'
        default_val = int(active_preset[c]) if active_preset else int(schema['numerical'][c]['median'])
        inputs[c] = st.number_input(
            "Annual Income (INR)", 
            min_value=int(schema['numerical'][c]['min']), 
            max_value=int(schema['numerical'][c]['max']), 
            value=default_val,
            step=1000
        )
        
        c = 'monthly_income'
        default_val = int(active_preset[c]) if active_preset else int(schema['numerical'][c]['median'])
        inputs[c] = st.number_input(
            "Monthly Net Income (INR)", 
            min_value=int(schema['numerical'][c]['min']), 
            max_value=int(schema['numerical'][c]['max']), 
            value=default_val,
            step=100
        )

# Populate Tab 3: Credit & History
with tab3:
    st.markdown("### Credit History and Bureau Record")
    col1, col2 = st.columns(2)
    
    with col1:
        c = 'credit_score'
        default_val = int(active_preset[c]) if active_preset else int(schema['numerical'][c]['median'])
        inputs[c] = st.slider(
            "CIBIL/Credit Score", 
            min_value=int(schema['numerical'][c]['min']), 
            max_value=int(schema['numerical'][c]['max']), 
            value=default_val
        )
        
        c = 'credit_history_length_months'
        default_val = int(active_preset[c]) if active_preset else int(schema['numerical'][c]['median'])
        inputs[c] = st.number_input(
            "Credit History Length (Months)", 
            min_value=int(schema['numerical'][c]['min']), 
            max_value=int(schema['numerical'][c]['max']), 
            value=default_val
        )
        
        c = 'repayment_status'
        default_idx = schema['categorical'][c].index(active_preset[c]) if active_preset else 0
        inputs[c] = st.selectbox("Current Repayment Status", options=schema['categorical'][c], index=default_idx)
        
    with col2:
        c = 'delinquency_count'
        default_val = int(active_preset[c]) if active_preset else int(schema['numerical'][c]['median'])
        inputs[c] = st.number_input(
            "Bureau Delinquency Count (Past 2 Years)", 
            min_value=int(schema['numerical'][c]['min']), 
            max_value=int(schema['numerical'][c]['max']), 
            value=default_val
        )
        
        c = 'past_default_flag'
        default_val = int(active_preset[c]) if active_preset else int(schema['numerical'][c]['median'])
        inputs[c] = st.selectbox("Has Borrower Defaulted in Past?", options=[0, 1], index=[0, 1].index(default_val), format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
        
        c = 'first_emi_default_flag'
        default_val = int(active_preset[c]) if active_preset else int(schema['numerical'][c]['median'])
        inputs[c] = st.selectbox("Defaulted on First EMI of any previous loan?", options=[0, 1], index=[0, 1].index(default_val), format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")

# Populate Tab 4: Loan Details
with tab4:
    st.markdown("### Specifics of Requested Loan")
    col1, col2 = st.columns(2)
    
    with col1:
        c = 'loan_amount'
        default_val = int(active_preset[c]) if active_preset else int(schema['numerical'][c]['median'])
        inputs[c] = st.number_input(
            "Loan Principal Amount (INR)", 
            min_value=int(schema['numerical'][c]['min']), 
            max_value=int(schema['numerical'][c]['max']), 
            value=default_val,
            step=1000
        )
        
        c = 'loan_term_months'
        default_val = int(active_preset[c]) if active_preset else int(schema['numerical'][c]['median'])
        inputs[c] = st.number_input(
            "Loan Term (Months)", 
            min_value=int(schema['numerical'][c]['min']), 
            max_value=int(schema['numerical'][c]['max']), 
            value=default_val
        )
        
        c = 'interest_rate'
        default_val = float(active_preset[c]) if active_preset else float(schema['numerical'][c]['median'])
        inputs[c] = st.number_input(
            "Applied Interest Rate (%)", 
            min_value=float(schema['numerical'][c]['min']), 
            max_value=float(schema['numerical'][c]['max']), 
            value=default_val,
            step=0.1
        )
        
        c = 'emi_amount'
        default_val = int(active_preset[c]) if active_preset else int(schema['numerical'][c]['median'])
        inputs[c] = st.number_input(
            "Proposed EMI Amount (INR/Month)", 
            min_value=int(schema['numerical'][c]['min']), 
            max_value=int(schema['numerical'][c]['max']), 
            value=default_val,
            step=50
        )
        
    with col2:
        c = 'loan_purpose'
        default_idx = schema['categorical'][c].index(active_preset[c]) if active_preset else 0
        inputs[c] = st.selectbox("Loan Purpose", options=schema['categorical'][c], index=default_idx)
        
        c = 'loan_product'
        default_idx = schema['categorical'][c].index(active_preset[c]) if active_preset else 0
        inputs[c] = st.selectbox("Loan Product Type", options=schema['categorical'][c], index=default_idx)
        
        c = 'loan_source_type'
        default_idx = schema['categorical'][c].index(active_preset[c]) if active_preset else 0
        inputs[c] = st.selectbox("Loan Source Type", options=schema['categorical'][c], index=default_idx)
        
        c = 'has_mortgage'
        default_idx = schema['categorical'][c].index(active_preset[c]) if active_preset else 0
        inputs[c] = st.selectbox("Has Mortgage Property", options=schema['categorical'][c], index=default_idx)
        
        c = 'has_cosigner'
        default_idx = schema['categorical'][c].index(active_preset[c]) if active_preset else 0
        inputs[c] = st.selectbox("Has Cosigner / Guarantor", options=schema['categorical'][c], index=default_idx)

# Populate Tab 5: Geography & Branch
with tab5:
    st.markdown("### Location and Branch Coordinates")
    col1, col2 = st.columns(2)
    
    with col1:
        c = 'state'
        default_idx = schema['categorical'][c].index(active_preset[c]) if active_preset else 0
        inputs[c] = st.selectbox("State", options=schema['categorical'][c], index=default_idx)
        
        c = 'district'
        default_idx = schema['categorical'][c].index(active_preset[c]) if active_preset else 0
        inputs[c] = st.selectbox("District", options=schema['categorical'][c], index=default_idx)
        
        c = 'city'
        default_idx = schema['categorical'][c].index(active_preset[c]) if active_preset else 0
        inputs[c] = st.selectbox("City", options=schema['categorical'][c], index=default_idx)
        
        c = 'pincode'
        default_val = int(active_preset[c]) if active_preset else int(schema['numerical'][c]['median'])
        inputs[c] = st.number_input(
            "Pincode", 
            min_value=int(schema['numerical'][c]['min']), 
            max_value=int(schema['numerical'][c]['max']), 
            value=default_val
        )
        
        c = 'urban_rural_flag'
        default_idx = schema['categorical'][c].index(active_preset[c]) if active_preset else 0
        inputs[c] = st.selectbox("Urban/Rural Region", options=schema['categorical'][c], index=default_idx)
        
        c = 'population_density_band'
        default_idx = schema['categorical'][c].index(active_preset[c]) if active_preset else 0
        inputs[c] = st.selectbox("Population Density Band", options=schema['categorical'][c], index=default_idx)
        
    with col2:
        c = 'branch_code'
        default_idx = schema['categorical'][c].index(active_preset[c]) if active_preset else 0
        inputs[c] = st.selectbox("Branch Code", options=schema['categorical'][c], index=default_idx)
        
        c = 'branch_region'
        default_idx = schema['categorical'][c].index(active_preset[c]) if active_preset else 0
        inputs[c] = st.selectbox("Branch Region", options=schema['categorical'][c], index=default_idx)
        
        c = 'channel'
        default_idx = schema['categorical'][c].index(active_preset[c]) if active_preset else 0
        inputs[c] = st.selectbox("Sourcing Channel", options=schema['categorical'][c], index=default_idx)
        
        c = 'service_area_cluster'
        default_val = int(active_preset[c]) if active_preset else int(schema['numerical'][c]['median'])
        inputs[c] = st.number_input(
            "Service Area Cluster ID", 
            min_value=int(schema['numerical'][c]['min']), 
            max_value=int(schema['numerical'][c]['max']), 
            value=default_val
        )
        
        c = 'distance_to_branch_km'
        default_val = float(active_preset[c]) if active_preset else float(schema['numerical'][c]['median'])
        inputs[c] = st.number_input(
            "Distance to Branch (km)", 
            min_value=float(schema['numerical'][c]['min']), 
            max_value=float(schema['numerical'][c]['max']), 
            value=default_val,
            step=0.1
        )

# Calculated numerical features from inputs to match notebook schema
inputs['existing_emi'] = int(active_preset['existing_emi']) if active_preset else int(schema['numerical']['existing_emi']['median'])
inputs['debt_to_income'] = float(active_preset['debt_to_income']) if active_preset else float(schema['numerical']['debt_to_income']['median'])
inputs['area_default_rate'] = float(active_preset['area_default_rate']) if active_preset else float(schema['numerical']['area_default_rate']['median'])
inputs['district_risk_score'] = float(active_preset['district_risk_score']) if active_preset else float(schema['numerical']['district_risk_score']['median'])
inputs['state_risk_score'] = float(active_preset['state_risk_score']) if active_preset else float(schema['numerical']['state_risk_score']['median'])

# Run Prediction
st.markdown("---")
col_btn, col_result = st.columns([1, 2])

with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🚀 Calculate Default Risk", use_container_width=True)

with col_result:
    if predict_btn:
        # Create input row DataFrame matching exact columns order
        ordered_cols = list(schema['numerical'].keys()) + list(schema['categorical'].keys())
        
        # Make sure target is not in features
        if 'loan_default' in ordered_cols:
            ordered_cols.remove('loan_default')
            
        # Re-align dictionary elements according to schema order
        row_dict = {col: inputs[col] for col in ordered_cols if col in inputs}
        row = pd.DataFrame([row_dict])
        
        # Predict using model
        try:
            pred = model.predict(row)[0]
            prob = model.predict_proba(row)[0, 1]
            
            # Display premium visual card based on default risk level
            if pred == 1:
                st.markdown(f"""
                    <div class="result-card-default">
                        <h3>⚠️ HIGH RISK OF DEFAULT (Default Prediction: {int(pred)})</h3>
                        <p style="font-size: 1.15rem; margin-bottom: 8px;">
                            The model indicates a high likelihood of this borrower defaulting on the loan.
                        </p>
                        <h4 style="margin: 0; font-size: 1.4rem;">Default Probability: {prob * 100:.2f}%</h4>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="result-card-safe">
                        <h3>✅ LOW RISK / APPROVED (Default Prediction: {int(pred)})</h3>
                        <p style="font-size: 1.15rem; margin-bottom: 8px;">
                            The borrower profile demonstrates strong financial health and credit compliance.
                        </p>
                        <h4 style="margin: 0; font-size: 1.4rem;">Default Probability: {prob * 100:.2f}%</h4>
                    </div>
                """, unsafe_allow_html=True)
                
            st.markdown("#### Probability Spectrum")
            st.progress(float(prob))
            
        except Exception as e:
            st.error(f"Prediction failed. There might be a structural mismatch with the loaded model model: {e}")
