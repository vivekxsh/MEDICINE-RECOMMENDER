import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load datasets
sym_des = pd.read_csv("datasets/symtoms_df.csv")
precautions = pd.read_csv("datasets/precautions_df.csv")
workout = pd.read_csv("datasets/workout_df.csv")
description = pd.read_csv("datasets/description.csv")
medications = pd.read_csv('datasets/medications.csv')
diets = pd.read_csv("datasets/diets.csv")

# Load model
svc = pickle.load(open('models/svc.pkl', 'rb'))

# Helper function
def helper(dis):
    desc = description[description['Disease'] == dis]['Description'].values[0]
    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values.flatten()
    med = medications[medications['Disease'] == dis]['Medication'].values.flatten()
    die = diets[diets['Disease'] == dis]['Diet'].values.flatten()
    wrkout = workout[workout['disease'] == dis]['workout'].values.flatten()
    return desc, pre, med, die, wrkout

symptoms_dict = { ... }  # Keep your existing symptoms_dict
diseases_list = { ... }  # Keep your existing diseases_list

def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]

# Streamlit app
st.title("Disease Prediction App")

# User input
symptoms_input = st.text_area("Enter symptoms (comma-separated):")

if st.button("Predict"):
    if symptoms_input == "":
        st.warning("Please enter symptoms.")
    else:
        user_symptoms = [s.strip() for s in symptoms_input.split(',')]
        predicted_disease = get_predicted_value(user_symptoms)
        dis_des, precautions_list, medications_list, rec_diet, workout_list = helper(predicted_disease)

        st.subheader(f"Predicted Disease: {predicted_disease}")
        st.write(f"Description: {dis_des}")
        
        st.write("Precautions:")
        for precaution in precautions_list:
            st.write(f"- {precaution}")

        st.write("Medications:")
        for med in medications_list:
            st.write(f"- {med}")

        st.write("Recommended Diet:")
        for diet in rec_diet:
            st.write(f"- {diet}")

        st.write("Recommended Workouts:")
        for workout in workout_list:
            st.write(f"- {workout}")

# About, Contact, Developer sections can be added similarly
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ("About", "Contact", "Developer"))

if page == "About":
    st.header("About This App")
    st.write("This app predicts diseases based on symptoms inputted by the user.")
elif page == "Contact":
    st.header("Contact Us")
    st.write("For inquiries, please reach out to us at contact@example.com.")
elif page == "Developer":
    st.header("Developer Information")
    st.write("Developed by Your Name.")

# Run the app
if __name__ == '__main__':
    st.run()
