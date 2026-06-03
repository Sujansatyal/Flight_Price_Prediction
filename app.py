import streamlit as st
import pandas as pd
import joblib

# Load Saved Objects

@st.cache_resource
def load_artifacts():
     st.write("Loading files...")
     model = joblib.load("Model/model.joblib")
     feature_names = joblib.load("Model/feature_names.joblib")
     ohe = joblib.load("Model/ohe.joblib")
     class_encoder = joblib.load("Model/class_encoder.joblib")
     stops_encoder = joblib.load("Model/stops_encoder.joblib")
     return model, feature_names, ohe, class_encoder, stops_encoder
model, feature_names, ohe, class_encoder, stops_encoder = load_artifacts()

# Page Config

st.set_page_config(
    page_title="Flight Price Prediction",
    page_icon="✈️",
    layout="centered"
)

st.title("✈️ Flight Price Prediction System")
st.write("Enter flight details to predict the ticket price.")

# User Inputs

airline = st.selectbox(
    "Airline",
    ['SpiceJet', 'AirAsia', 'Vistara', 'GO_FIRST', 'Indigo', 'Air_India']
)

source_city = st.selectbox(
    "Source City",
    ['Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Hyderabad', 'Chennai']
)

departure_time = st.selectbox(
    "Departure Time",
    ['Evening', 'Early_Morning', 'Morning',
     'Afternoon', 'Night', 'Late_Night']
)

stops = st.selectbox(
    "Stops",
    ['zero', 'one', 'two_or_more']
)

arrival_time = st.selectbox(
    "Arrival Time",
    ['Night', 'Morning', 'Early_Morning',
     'Afternoon', 'Evening', 'Late_Night']
)

destination_city = st.selectbox(
    "Destination City",
    ['Mumbai', 'Bangalore', 'Kolkata',
     'Hyderabad', 'Chennai', 'Delhi']
)

travel_class = st.selectbox(
    "Class",
    ['Economy', 'Business']
)

duration = st.number_input(
    "Duration (Hours)",
    min_value=0.0,
    step=0.1
)

days_left = st.number_input(
    "Days Left Before Departure",
    min_value=1,
    step=1
)

# Prediction

if st.button("Predict Price"):

    class_encoded = class_encoder.transform([travel_class])[0]
    stops_encoded = stops_encoder.transform([stops])[0]

    input_df = pd.DataFrame({
        "airline": [airline],
        "source_city": [source_city],
        "departure_time": [departure_time],
        "stops": [stops_encoded],
        "arrival_time": [arrival_time],
        "destination_city": [destination_city],
        "class": [class_encoded],
        "duration": [duration],
        "days_left": [days_left]
    })

    categorical_cols = [
        "airline",
        "source_city",
        "departure_time",
        "arrival_time",
        "destination_city"
    ]

    encoded_features = ohe.transform(input_df[categorical_cols])

    encoded_df = pd.DataFrame(
        encoded_features,
        columns=ohe.get_feature_names_out(categorical_cols)
    )

    final_input = pd.concat(
        [
            encoded_df.reset_index(drop=True),
            input_df[["stops", "class", "duration", "days_left"]]
                .reset_index(drop=True)
        ],
        axis=1
    )
    final_input = final_input.reindex(
    columns=feature_names,
    fill_value=0
)

    prediction = model.predict(final_input)

    st.success(
        f"Estimated Flight Price: ₹ {prediction[0]:,.2f}"
    )