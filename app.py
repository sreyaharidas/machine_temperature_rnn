import streamlit as st
import numpy as np
import tensorflow as tf

# Load trained RNN model
model = tf.keras.models.load_model("machine_temperature_rnn.keras")

# Title
st.title("Machine Temperature Predictor")

# Previous Timestamp 1
temperature1 = st.number_input(
    "Previous Timestamp 1 - Temperature",
    value=81.0
)

vibration1 = st.number_input(
    "Previous Timestamp 1 - Vibration",
    value=3.5
)

# Previous Timestamp 2
temperature2 = st.number_input(
    "Previous Timestamp 2 - Temperature",
    value=83.0
)

vibration2 = st.number_input(
    "Previous Timestamp 2 - Vibration",
    value=3.6
)

# Prediction button
if st.button("Predict Next Temperature"):

    # Create input in RNN format:
    # (samples, time steps, features)
    new_data = np.array([
        [
            [temperature1, vibration1],
            [temperature2, vibration2]
        ]
    ], dtype=np.float32)

    # Make prediction
    prediction = model.predict(new_data, verbose=0)

    next_temperature = prediction[0][0]

    # Display result
    st.success(
        f"Predicted Next Machine Temperature: {next_temperature:.2f} °C"
    )
