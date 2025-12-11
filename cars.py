import pickle
import streamlit as st
import pandas as pd
import lightgbm as lgb
import numpy as np
import joblib
from sklearn.preprocessing import RobustScaler


cars_df = pd.read_csv('cars24.csv')

st.title('Car Data Explorer')
st.write('This application allows you to explore car data.')

st.header('Data Set Overview')
st.write(f'The data set contains {cars_df.shape[0]} rows and {cars_df.shape[1]} columns.')
st.write('Here are a sample of 10 rows of the data set:')
st.dataframe(cars_df.sample(10))

with open('lgbm_model_final.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler_joblib.pkl', 'rb') as file:
    scaler = joblib.load(file)

st.header('Model Prediction')
st.write('Input features to get a prediction from the loaded model.')
age=st.number_input('Enter age:', value=18)
income=st.number_input('Enter income:', value=50000)
business_value=st.number_input('Enter business value:', value=100000)
quart_rating=st.number_input('Enter quarterly rating:', value=3)

input_data = np.array([[age, income, business_value, quart_rating]])
input_data_scaled = scaler.transform(input_data)
st.write(input_data_scaled)


test=np.array([ 4.28571429e-01,  0.00000000e+00,  2.00000000e+00,  4.65597858e-02,
        1.80000000e+01,  1.00000000e+01,  2.01900000e+03,  1.00000000e+00,
        1.00000000e+00, -1.83878424e-01, -3.42857143e-01,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]).reshape(1,-1)
st.write(model.predict(test))

# query=np.array([input_data_scaled[0],  0.00000000e+00,  2.00000000e+00,  input_data_scaled[1],
#         1.80000000e+01,  1.00000000e+01,  2.01900000e+03,  1.00000000e+00,
#         1.00000000e+00, input_data_scaled[2], input_data_scaled[3],  0.00000000e+00,
#         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00,
#         0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]).reshape(1,-1)