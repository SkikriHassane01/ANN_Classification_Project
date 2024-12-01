import pandas as pd
import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from tensorflow.keras.models import load_model
import pickle
 
 # you need to ensure that the version of the scikit-learn is the same that you use to pickle the files for my case i use 1.2.2 
# Load the trained model
model = load_model('model.h5')

# Load encoder and scaler 
with open('LB_gender.pkl', 'rb') as file:
    LB_gender = pickle.load(file)

with open('OHE_geo.pkl', 'rb') as file:
    OHE_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Define the streamlit app
st.title('Customer Churn Prediction')

# Add input fields for user to enter data
geography = st.selectbox('Geography', ['France', 'Spain', 'Germany'])
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.number_input('Age', min_value=18, max_value=100)
tenure = st.number_input('Tenure', min_value=0, max_value=10)
balance = st.number_input('Balance')
num_of_products = st.number_input('Number of Products', min_value=1, max_value=4)
has_crdt_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is active member ', [0, 1])
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_crdt_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Preprocess the input data
# Transform Gender using LabelEncoder
input_data['Gender'] = LB_gender.transform([input_data['Gender']])[0]

# Encode Geography using OneHotEncoder
geo_encoded = OHE_geo.transform(input_data[['Geography']])
geoencoded_df = pd.DataFrame(geo_encoded, columns=OHE_geo.get_feature_names_out())

# Concatenate the encoded geography with the rest of the data
input_data = pd.concat([input_data.drop('Geography', axis=1), geoencoded_df], axis=1)
# print("Scaler columns:", scaler.feature_names_in_)

# Define the expected column order based on the model training
expected_column_order = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
                         'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Gender', 
                         'Geography_France', 'Geography_Germany', 'Geography_Spain']

# Reorder the columns in input_data to match the expected order
input_data = input_data[expected_column_order]

# Apply the scaler transformation to the input data
input_data_scaled = scaler.transform(input_data)

# Make a prediction using the model
prediction = model.predict(input_data_scaled)

# Display the prediction result
if st.button('Predict'):
    st.write(f"Churn Probability: {prediction[0][0].round(2)}")
    
    if prediction[0][0] > 0.5:
        st.write('The customer is likely to churn.')
    else:
        st.write('The customer is unlikely to churn.')
