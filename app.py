import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

# --- PAGE CONFIGURATION (Must be the first Streamlit command) ---
st.set_page_config(page_title="Bengaluru Real Estate Prediction", page_icon="🏡", layout="wide")

# --- 1. LOAD THE AI ASSETS ---
@st.cache_resource
def load_artifacts():
    with open('bangalore_home_prices_model.pickle', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pickle', 'rb') as f:
        scaler = pickle.load(f)
    with open('columns.json', 'r') as f:
        cols = json.load(f)['data_columns']
    return model, scaler, cols

model, scaler, data_columns = load_artifacts()

# Extract just the location names for the dropdown menu
locations = [col.replace('location_', '') for col in data_columns if col.startswith('location_')]

# --- 2. BUILD THE INTERACTIVE UI ---
st.title("🏡 Bengaluru Real Estate Price Predictor")
st.markdown("### *AI-Powered Valuation Engine (Optimized Ridge Regression)*")
st.markdown("---")

# Create Interactive Tabs
tab1, tab2 = st.tabs(["🔮 Live Price Predictor", "📊 Under The Hood (AI Metrics)"])

with tab1:
    st.subheader("Enter Property Details")
    
    # Use columns to make the layout wider and more interactive
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_sqft = st.number_input("Total Square Feet", min_value=300, max_value=10000, value=1000, step=100)
    with col2:
        bhk = st.slider("Number of Bedrooms (BHK)", min_value=1, max_value=10, value=2)
    with col3:
        bath = st.slider("Number of Bathrooms", min_value=1, max_value=10, value=2)
        
    # Full width dropdown for location
    location = st.selectbox("Select Neighborhood Location", locations)

    st.markdown("---")
    
    # --- 3. PREDICTION ENGINE ---
    # Centered predict button
    if st.button("Calculate Property Value", type="primary", use_container_width=True):
        with st.spinner('Running AI Model...'):
            # Data Formatting
            input_df = pd.DataFrame(columns=data_columns)
            input_df.loc[0] = 0  
            
            input_df['total_sqft'] = total_sqft
            input_df['bath'] = bath
            input_df['BHK'] = bhk
            
            loc_column = f'location_{location}'
            if loc_column in data_columns:
                input_df[loc_column] = 1
                
            # Scale and Predict
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            
            # Interactive Results Display
            st.success("Analysis Complete!")
            
            # Use Streamlit's built-in metric layout for a clean look
            res_col1, res_col2, res_col3 = st.columns([1, 2, 1])
            with res_col2:
                st.metric(label="Estimated Market Value", value=f"₹ {round(prediction, 2)} Lakhs", delta="80.4% AI Accuracy Ceiling")
            
            with st.expander("⚠️ View Disclaimer & Limitations"):
                st.write("""
                *This model calculates price based strictly on location, square footage, and room counts. It cannot account for unquantifiable human factors like interior design quality, exact floor level, or current seller urgency.*
                """)

with tab2:
    st.subheader("Model Training Architecture & Performance")
    st.write("This application runs on a tuned Ridge Regression model trained on 13,000+ properties.")
    
    met_col1, met_col2 = st.columns(2)
    with met_col1:
        st.info("**Baseline Model (Week 2)**\n* R-Squared: 49.39%\n* Mean Error: 38.34 Lakhs")
    with met_col2:
        st.success("**Optimized Model (Week 3)**\n* R-Squared: 80.40%\n* Mean Error: 16.72 Lakhs")
        
    st.markdown("#### Optimization Techniques Applied:")
    st.code("""
    1. Feature Engineering: price_per_sqft interaction variable
    2. Anomaly Filtering: Standard deviation filtering per location
    3. Tuning: GridSearchCV with 5-Fold Cross Validation
    4. Regularization: Ridge Penalty (alpha=1.0)
    """, language="text")