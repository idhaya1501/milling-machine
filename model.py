import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pickle

# Load dataset
df = pd.read_csv("machine_failure_data.csv")

# Define features and labels
features = ['Type', 'Air temperature [K]', 'Process temperature [K]', 
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
labels = 'Failure_Type'  # Assuming the dataset has a categorical failure label

X = df[features].values
y = df[labels].values

# Normalize input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler for use in app.py
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Reshape for LSTM (samples, time steps, features)
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Convert labels to categorical if necessary
num_classes = len(np.unique(y))  # Determine number of unique failure types
y_categorical = tf.keras.utils.to_categorical(y, num_classes=num_classes)

# Build LSTM Model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(1, len(features))),
    tf.keras.layers.LSTM(32, return_sequences=False),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')  # Multi-class classification
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_scaled, y_categorical, epochs=20, batch_size=16, validation_split=0.2)

# Save trained model
model.save("lstm_failure_model.h5")
