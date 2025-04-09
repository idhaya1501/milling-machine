from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import pickle
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained LSTM model
try:
    model = load_model("lstm_failure_model.h5")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load the scaler for input normalization
try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    print(f"Loaded object type: {type(scaler)}")  # Check what type of object is loaded

    if not hasattr(scaler, "transform"):  # Ensure it's a valid Scaler
        raise ValueError("Loaded object is not a valid StandardScaler.")

    print("Scaler loaded successfully.")
except Exception as e:
    print(f"Error loading scaler: {e}")

# Failure reason mapping
reason_mapping = {
    0: 'HDF',  # Tool Wear Failure
    1: 'OSF',  # Heat Dissipation Failure
    2: 'PWF',  # Power Failure
    3: 'RNF',  # Overstrain Failure
    4: 'TWF',  # Random Failure
    5: 'Normal'  # No Failure
}

# Email Configuration
EMAIL_SENDER = "sidhayaventhan@gmail.com"
EMAIL_RECEIVER = "sidhayaventhan1@gmail.com"
EMAIL_PASSWORD = "bjog svqm vtxa glsg"

# Email Notification Function
def send_email(failure_reason):
    if failure_reason == "Normal":
        return  # Don't send an email if there's no failure

    email_body = f"""
    <html>
    <body>
        <h2 style="color: red;">MACHINE FAILURE DETECTED !!</h2>
        <p><h3>Failure Reason: <strong>{failure_reason}</strong></h3></p>
    </body>
    </html>
    """

    message = MIMEMultipart()
    message["Subject"] = "MACHINE FAILURE WARNING"
    message["From"] = EMAIL_SENDER
    message["To"] = EMAIL_RECEIVER

    message.attach(MIMEText(email_body, "html"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.ehlo()
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, message.as_string())
        server.quit()
        print("Email sent successfully.")
    except Exception as e:
        print(f"Error sending email: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/index1')
def index1():
    return render_template('index1.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/monitoring')
def monitoring():
    return render_template('monitoring.html')

@app.route('/predict_ML', methods=['POST'])
def predict_ML():
    try:
        if model is None or scaler is None:
            return render_template('index.html', prediction="Model or Scaler Not Loaded Correctly")

        # Get input values from the form and convert to float
        Type = float(request.form['Type'])
        Air_temperature = float(request.form['Air_temperature'])
        Process_temperature = float(request.form['Process_temperature'])
        Rotational_speed = float(request.form['Rotational_speed'])
        Torque = float(request.form['Torque'])
        Tool_wear = float(request.form['Tool_wear'])

        # Prepare input data
        user_input = np.array([[Type, Air_temperature, Process_temperature, 
                               Rotational_speed, Torque, Tool_wear]])
        
        # Apply scaler transformation
        user_input = scaler.transform(user_input)  # Normalize input
        user_input = user_input.reshape(1, 1, user_input.shape[1])  # Reshape for LSTM input

        # Make prediction
        prediction = model.predict(user_input)
        predicted_class = np.argmax(prediction)  # Get the class with highest probability
        failure_reason = reason_mapping.get(predicted_class, "Unknown Failure")
      
        # Send email if failure is detected
        send_email(failure_reason)
        
        # Return prediction result
        return render_template('index.html', prediction=failure_reason,
                               Type=Type, Air_temperature=Air_temperature,
                               Process_temperature=Process_temperature,
                               Rotational_speed=Rotational_speed,
                               Torque=Torque, Tool_wear=Tool_wear)
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")


@app.route('/Control', methods=['POST'])
def predict_Control():
    try:
        if model is None or scaler is None:
            return render_template('index1.html', prediction="Model or Scaler Not Loaded Correctly")

        # Get input values from the form and convert to float
        Type = int(request.form['Type'])
        Air_temperature = float(request.form['Air_temperature'])
        Process_temperature = float(request.form['Process_temperature'])
        Rotational_speed = float(request.form['Rotational_speed'])
        Torque = float(request.form['Torque'])
        Tool_wear = float(request.form['Tool_wear'])

        # Prepare input data
        user_input = np.array([[Type, Air_temperature, Process_temperature, 
                               Rotational_speed, Torque, Tool_wear]])
        
        # Apply scaler transformation
        user_input = scaler.transform(user_input)  # Normalize input
        user_input = user_input.reshape(1, 1, user_input.shape[1])  # Reshape for LSTM input

        # Make prediction
        prediction = model.predict(user_input)
        predicted_class = np.argmax(prediction)  # Get the class with highest probability
        failure_reason = reason_mapping.get(predicted_class, "Unknown Failure")


        # Return prediction result
        return render_template('index1.html', prediction=failure_reason,
                               Type=Type, Air_temperature=Air_temperature,
                               Process_temperature=Process_temperature,
                               Rotational_speed=Rotational_speed,
                               Torque=Torque, Tool_wear=Tool_wear)
    except Exception as e:
        return render_template('index1.html', prediction=f"Error: {str(e)}")



if __name__ == '__main__':
    app.run(debug=True)
