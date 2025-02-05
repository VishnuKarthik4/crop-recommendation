#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import pickle


# # Load the Model

# In[3]:


with open('CropRec.pkl', 'rb') as file:
    model = pickle.load(file)


# In[7]:


import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load model (assuming it's already trained and loaded as `model`)
# model = ...

st.title("Crop Recommendation App")

st.sidebar.header("Soil and Climate Data")

# Function for user input
def user_input_features():
    N = st.sidebar.number_input('Nitrogen (N)', min_value=0)
    P = st.sidebar.number_input('Phosphorus (P)', min_value=0)
    K = st.sidebar.number_input('Potassium (K)', min_value=0)
    temperature = st.sidebar.number_input('Temperature (°C)', min_value=0.0)
    humidity = st.sidebar.number_input('Humidity (%)', min_value=0.0, max_value=100.0)
    ph = st.sidebar.number_input('Soil pH', min_value=0.0, max_value=14.0)
    rainfall = st.sidebar.number_input('Rainfall (mm)', min_value=0.0)
    
    data = {
        'N': N,
        'P': P,
        'K': K,
        'temperature': temperature,
        'humidity': humidity,
        'ph': ph,
        'rainfall': rainfall
    }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader("Input Data")
st.write(df)

if st.button('Predict'):
    # Assuming model is trained and loaded
    prediction = model.predict(df)
    prediction_proba = model.predict_proba(df)

    st.subheader("Crop Recommendation")
    st.write(f"Recommended Crop: {prediction[0]}")

    st.subheader("Prediction Probability")

    # Prediction probability plot (horizontal bar chart with increased thickness)
    prob_df = pd.DataFrame(prediction_proba, columns=model.classes_).T
    prob_df.columns = ['Probability']
    prob_df['Crops'] = prob_df.index

    # Plot the prediction probabilities using Plotly (Horizontal Bar with increased width)
    prob_fig = px.bar(prob_df, x='Probability', y='Crops', orientation='h', title="Prediction Probability for Crops")
    
    # Adjust the height of the chart to increase bar thickness
    prob_fig.update_layout(
        height=600,  # Adjust the height to make bars thicker
        bargap=0.2   # Adjust the gap between bars (less gap makes bars thicker)
    )
    
    st.plotly_chart(prob_fig)

    # Plot input features using Plotly (Horizontal Bar)
    st.subheader("Input Feature Distribution")

    # Create a horizontal bar chart for input features
    input_fig = go.Figure(go.Bar(
        x=df.values[0],
        y=df.columns,
        orientation='h',
        marker=dict(color='rgba(0, 123, 255, 0.6)')
    ))
    input_fig.update_layout(
        title="Soil and Climate Features",
        xaxis_title="Value",
        yaxis_title="Feature"
    )

    st.plotly_chart(input_fig)


# In[ ]:




