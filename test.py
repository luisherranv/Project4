import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# Load the model and scaler
model = tf.keras.models.load_model('SpotifyModelOptimized_5.h5')
scaler = joblib.load('scaler_5.pkl')
with open('training_columns 2.txt', 'r') as f:
    training_columns = [line.strip() for line in f]

# Read test data from CSV
test_data = pd.read_csv('most_pop.csv')

# Function to preprocess and predict
def predict(test_data):
    # Ensure the test_data is in the expected format
    dummied_input_data = pd.get_dummies(test_data, columns=['genre'])
    dummied_input_data = dummied_input_data.reindex(columns=training_columns, fill_value=0)
    scaled_input_data = scaler.transform(dummied_input_data)

    # Make prediction
    predictions = model.predict(scaled_input_data)
    return predictions

# Make predictions on test data
results = predict(test_data)

# Print results
for i, probabilities in enumerate(results):
    chosen_class = np.argmax(probabilities)
    class_labels = ['1stQ', '2ndQ', '3rdQ', '4thQ']
    print(f"Data Point {i+1}:")
    print(f"Probabilities: {probabilities}")
    print(f"Chosen Class: {class_labels[chosen_class]}")
    print()
