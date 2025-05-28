🧠 Stroke Prediction Web App using Machine Learning
This is a web-based stroke prediction application developed using Flask and powered by a Machine Learning model trained to detect the likelihood of a person suffering a stroke based on medical and demographic features.

🚀 Project Overview
The goal of this project is to predict whether a patient is at high or low risk of stroke. The system uses a trained classification model and provides a simple web interface for users to input data and receive predictions.

🛠 Tech Stack
Frontend: HTML (via Jinja2 templates in Flask)

Backend: Python, Flask

Model: Scikit-learn model loaded via joblib

Deployment: Localhost (Flask development server)

📁 File Structure

├── app.py                # Main Flask application
├── stroke_model.pkl      # Pre-trained ML model
├── scaler.pkl            # Scaler used for feature normalization
├── features.pkl          # Feature names used in model training (optional)
├── templates/
│   └── index.html        # HTML template for input form and results

🧪 Features Used
The model uses the following features for prediction:

Age

Hypertension (binary)

Heart Disease (binary)

Average Glucose Level

Body Mass Index (BMI)

Gender

Marital Status

Work Type

Residence Type

Smoking Status

Categorical features are one-hot encoded before being passed into the model. The input is then scaled using a pre-fitted StandardScaler.

🌐 How It Works
User visits the home page and fills out the form.

Form data is processed and encoded.

The input is scaled and passed to the model.

The model returns a prediction (High or Low risk).

The result is displayed on the page.
