import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title="Real Estate Price Predictor", layout="wide")

st.title("🏠 Bengaluru House Price Prediction")
st.markdown("Enter property details to get estimated price")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("house_price_model.pkl")

try:
    data = load_model()
    model = data['model']
    scaler = data['scaler']
    label_encoders = data['label_encoders']

    # Get actual unique values from encoders
    location_options = label_encoders['location'].classes_.tolist() if 'location' in label_encoders else ["Whitefield", "Koramangala", "Indiranagar", "JP Nagar", "Electronic City"]
    area_type_options = label_encoders['area_type'].classes_.tolist() if 'area_type' in label_encoders else ["Super built-up", "Built-up", "Plot", "Carpet"]
    availability_options = label_encoders['availability'].classes_.tolist() if 'availability' in label_encoders else ["Ready To Move", "Immediate Possession"]
    society_options = label_encoders['society'].classes_.tolist() if 'society' in label_encoders else ["None"]

    # Input form
    col1, col2 = st.columns(2)

    with col1:
        location = st.selectbox("Location", location_options)
        bhk = st.slider("BHK", 1, 5, 2)
        total_sqft = st.number_input("Total Sqft", min_value=300, max_value=10000, value=1200)
        bath = st.slider("Bathrooms", 1, 5, 2)

    with col2:
        balcony = st.slider("Balconies", 0, 3, 1)
        area_type = st.selectbox("Area Type", area_type_options)
        availability = st.selectbox("Availability", availability_options)
        society = st.selectbox("Society", society_options)

    if st.button("Predict Price"):
        # Calculate engineered features
        sqft_per_bath = total_sqft / (bath + 1)
        price_per_sqft = 0

        # Encode categorical variables
        location_enc = label_encoders['location'].transform([location])[0] if 'location' in label_encoders else 0
        area_type_enc = label_encoders['area_type'].transform([area_type])[0] if 'area_type' in label_encoders else 0
        availability_enc = label_encoders['availability'].transform([availability])[0] if 'availability' in label_encoders else 0
        society_enc = label_encoders['society'].transform([society])[0] if 'society' in label_encoders else 0

        # Create input array in EXACT order: location, size, total_sqft, bath, balcony, area_type, availability, society, price_per_sqft, sqft_per_bath
        input_array = np.array([[location_enc, bhk, total_sqft, bath, balcony, area_type_enc, availability_enc, society_enc, price_per_sqft, sqft_per_bath]])

        # Scale and predict
        input_scaled = scaler.transform(input_array)
        log_price = model.predict(input_scaled)[0]
        predicted_price = np.expm1(log_price)

        st.success(f"Estimated Price: ₹{predicted_price:,.0f}")

except FileNotFoundError:
    st.error("Model not found. Please train the model first by running: python real_estate_project1_starter.py")
except Exception as e:
    st.error(f"Error: {str(e)}")
