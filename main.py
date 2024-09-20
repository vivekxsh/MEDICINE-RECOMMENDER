from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import pickle
import os

# Initialize Flask app
app = Flask(__name__)

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
def get_helper_data(dis):
    desc = " ".join(description[description['Disease'] == dis]['Description'].values)
    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values.flatten().tolist()
    med = medications[medications['Disease'] == dis]['Medication'].tolist()
    die = diets[diets['Disease'] == dis]['Diet'].tolist()
    wrkout = workout[workout['disease'] == dis]['workout'].tolist()
    return desc, pre, med, die, wrkout

# Mapping of symptoms and diseases
symptoms_dict = { ... }  # (Keep your symptoms_dict here)
diseases_list = { ... }  # (Keep your diseases_list here)

# Model prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]

# Routes
@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')
        if symptoms == "Symptoms":
            message = "Please either write symptoms or you have written misspelled symptoms."
            return render_template('index.html', message=message)
        
        user_symptoms = [s.strip() for s in symptoms.split(',')]
        predicted_disease = get_predicted_value(user_symptoms)
        dis_des, precautions, medications, rec_diet, workout = get_helper_data(predicted_disease)

        return render_template('index2.html', predicted_disease=predicted_disease, 
                               dis_des=dis_des, precautions=precautions, 
                               medications=medications, my_diet=rec_diet, 
                               workout=workout)

    return render_template('index2.html')

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/developer')
def developer():
    return render_template("developer.html")

@app.route('/blog')
def blog():
    return render_template("blog.html")

@app.route('/WEB_UIKITS')
def web_uikits():
    return render_template("WEB_UIKITS.html")

@app.route('/fillData')
def fill_data():
    return render_template("index.js")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use environment variable or default to 5000
    app.run(host="0.0.0.0", port=port, debug=True)
