from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load('stroke_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        age = float(request.form['age'])
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        avg_glucose = float(request.form['avg_glucose_level'])
        bmi = float(request.form['bmi'])

        # Encode categorical features
        gender = request.form['gender']
        ever_married = request.form['ever_married']
        work_type = request.form['work_type']
        residence_type = request.form['residence_type']
        smoking_status = request.form['smoking_status']

        # One-hot encoding
        gender_male = 1 if gender == 'Male' else 0
        gender_other = 1 if gender == 'Other' else 0
        ever_married_yes = 1 if ever_married == 'Yes' else 0

        work_type_dict = {
            'Private': [1, 0, 0, 0],
            'Self-employed': [0, 1, 0, 0],
            'Children': [0, 0, 1, 0],
            'Never worked': [0, 0, 0, 1],
            'Govt_job': [0, 0, 0, 0]  # default
        }
        work_type_encoded = work_type_dict.get(work_type, [0, 0, 0, 0])

        residence_type_urban = 1 if residence_type == 'Urban' else 0

        smoking_dict = {
            'formerly smoked': [1, 0, 0],
            'never smoked': [0, 1, 0],
            'smokes': [0, 0, 1],
            'Unknown': [0, 0, 0]
        }
        smoking_encoded = smoking_dict.get(smoking_status, [0, 0, 0])

        # Combine all inputs
        features = [
            age, hypertension, heart_disease, avg_glucose, bmi,
            gender_male, gender_other, ever_married_yes,
            *work_type_encoded, residence_type_urban,
            *smoking_encoded
        ]

        # Scale input
        scaled_input = scaler.transform([features])

        # Predict
        prediction = model.predict(scaled_input)[0]
        result = 'High Risk of Stroke' if prediction == 1 else 'Low Risk of Stroke'

        return render_template('index.html', prediction=result)

    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
