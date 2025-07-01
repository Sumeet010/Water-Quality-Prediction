import pandas as pd
import numpy as np
import joblib
import streamlit as st

# Loading model and model columns
model = joblib.load("pollution_model.pkl")
model_cols = joblib.load("model_columns.pkl")


thresholds = {
    'BSK5': 3,      # BSK5 should be <= 3
    'O2': 5,        # O2 should be > 5 (safe)
    'NO3': 10,      # NO3 should be <= 10 (safe)
    'NO2': 0.1,       # NO2 should be <= 0.1
    'SO4': 250,     # SO4 should be <= 250
    'PO4': 0.3,     # PO4 should be <= 0.1
    'CL': 250,      # CL should be <= 250
}

# UI
st.title("Water Quality Pollution Predictor")
st.write("Predict the water pollutants based on Year and Station ID")

year_input = st.number_input("Enter Year", min_value=2022, max_value=2100, value=2022 , key="year_input")
station_id = st.text_input("Enter the Station ID", value='1' , key="station_id")
#st.write("Model expects these columns:", model_cols)


if st.button("Predict"):
    if not station_id.strip():
        st.warning("Please enter the station ID")
    else:
        input_df = pd.DataFrame({'year': [year_input], 'id': [station_id]})
        input_encoded = pd.get_dummies(input_df, columns=['id'])

        for col in model_cols:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[model_cols]

        predicted_pollutants = model.predict(input_encoded)[0]

        pollutants = ['BSK5','O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL',]

        st.subheader(f"Predicted pollutant levels for station '{station_id}' in {year_input}:")
        for p, val in zip(pollutants, predicted_pollutants):
            st.write(f"{p}: {val:.2f}")


        safe = True
        for i, pollutant in enumerate(pollutants):
            val = predicted_pollutants[i]
            if pollutant in thresholds:
                if pollutant == 'O2':
                    if val < thresholds[pollutant]:
                        safe = False
                        break
                else:
                    if val > thresholds[pollutant]:
                        safe = False
                        break

        if safe:
            st.success("✅ The water is safe to drink.")
        else:
            st.error("❌ The water is NOT safe to drink.")

# st.write("Encoded input passed to model:", input_encoded)
import pandas as pd
import numpy as np
import joblib
import streamlit as st

# Loading model and model columns
model = joblib.load("pollution_model.pkl")
model_cols = joblib.load("model_columns.pkl")


thresholds = {
    'BSK5': 3,      # BSK5 should be <= 3
    'O2': 5,        # O2 should be > 5 (safe)
    'NO3': 10,      # NO3 should be <= 10 (safe)
    'NO2': 0.1,       # NO2 should be <= 0.1
    'SO4': 250,     # SO4 should be <= 250
    'PO4': 0.3,     # PO4 should be <= 0.1
    'CL': 250,      # CL should be <= 250
}

# UI
st.title("Water Quality Pollution Predictor")
st.write("Predict the water pollutants based on Year and Station ID")

year_input = st.number_input("Enter Year", min_value=2022, max_value=2100, value=2022)
station_id = st.text_input("Enter the Station ID", value='1')
#st.write("Model expects these columns:", model_cols)


if st.button("Predict"):
    if not station_id.strip():
        st.warning("Please enter the station ID")
    else:
        input_df = pd.DataFrame({'year': [year_input], 'id': [station_id]})
        input_encoded = pd.get_dummies(input_df, columns=['id'])

        for col in model_cols:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[model_cols]

        predicted_pollutants = model.predict(input_encoded)[0]

        pollutants = ['BSK5','O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL',]

        st.subheader(f"Predicted pollutant levels for station '{station_id}' in {year_input}:")
        for p, val in zip(pollutants, predicted_pollutants):
            st.write(f"{p}: {val:.2f}")


        safe = True
        for i, pollutant in enumerate(pollutants):
            val = predicted_pollutants[i]
            if pollutant in thresholds:
                if pollutant == 'O2':
                    if val < thresholds[pollutant]:
                        safe = False
                        break
                else:
                    if val > thresholds[pollutant]:
                        safe = False
                        break

        if safe:
            st.success("✅ The water is safe to drink.")
        else:
            st.error("❌ The water is NOT safe to drink.")

# st.write("Encoded input passed to model:", input_encoded)