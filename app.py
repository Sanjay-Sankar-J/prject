import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Load the model and scaler
model = load_model('deep_learning_model(2).h5')
scaler = joblib.load('scaler(2).pkl')

# Streamlit UI
st.title('Classification Prediction using Deep Learning')

# Top 10 selected features
selected_features = ['total_time3', 'gmrt_in_air7', 'mean_gmrt7', 'mean_speed_in_air7',
                     'paper_time9', 'total_time9', 'air_time16', 'mean_gmrt17',
                     'disp_index22', 'disp_index23']

# Inputs for the 10 features using sliders
features = []
for feature_name in selected_features:
    feature = st.slider(f'{feature_name}', min_value=0.0, max_value=100.0, value=0.0)
    features.append(feature)

# Convert the inputs to a numpy array
input_data = np.array(features).reshape(1, -1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict
if st.button('Predict'):
    prediction = model.predict(input_data_scaled)
    class_name = 'Class 1' if prediction[0][0] > 0.5 else 'Class 0'
    st.write(f'Prediction: {class_name}')
