from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
import warnings


warnings.filterwarnings('ignore')

app = Flask(__name__, template_folder='templates')

# Load the model (adjust the path as necessary)
model = tf.keras.models.load_model('SpotifyModelOptimized_5.h5')
scaler = joblib.load('scaler_5.pkl')
with open('training_columns 2.txt', 'r') as f:
    training_columns = [line.strip() for line in f]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        genre = request.form['genre']
        danceability = float(request.form['danceability'])
        energy = float(request.form['energy'])
        key = float(request.form['key'])
        loudness = float(request.form['loudness'])
        speechiness = float(request.form['speechiness'])
        acousticness = float(request.form['acousticness'])
        instrumentalness = float(request.form['instrumentalness'])
        liveness = float(request.form['liveness'])
        valence = float(request.form['valence'])
        tempo = float(request.form['tempo'])
        time_signature = float(request.form['time_signature'])
        duration_ms = float(request.form['duration_ms'])
        artist_popularity = float(request.form['artist_popularity'])

        input_data = pd.DataFrame({
                'genre': [genre],
                'danceability': [danceability],
                'energy': [energy],
                'key': [key],
                'loudness': [loudness],
                'speechiness': [speechiness],
                'acousticness': [acousticness],
                'instrumentalness': [instrumentalness],
                'liveness': [liveness],
                'valence': [valence],
                'tempo': [tempo],
                'time_signature': [time_signature],
                'duration_ms': [duration_ms],
                'artist_popularity': [artist_popularity]
            })
        
        print("Input Data DataFrame:", input_data)

        dummied_input_data = pd.get_dummies(input_data, columns=['genre'])
        dummied_input_data = dummied_input_data.reindex(columns=training_columns, fill_value=0)
        scaled_input_data =  scaler.transform(dummied_input_data)

        prediction = model.predict(scaled_input_data)
        probabilities = prediction[0]
        chosen_class = np.argmax(probabilities)
        class_labels = ['1stQ', '2ndQ', '3rdQ', '4thQ'] 
        
        return render_template('index.html', 
                            probabilities=probabilities,
                            chosen_class=class_labels[chosen_class],
                            prediction_text=f'Chosen Class: {class_labels[chosen_class]}',
                            enumerate=enumerate)
    return 'Method not allowed', 405

if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=8080)
    app.run(debug=True, port=5001)






