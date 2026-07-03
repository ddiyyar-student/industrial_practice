from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np
import os

app = Flask(__name__, static_folder='static')


BASE = os.path.dirname(__file__)
model         = joblib.load(os.path.join(BASE, 'model.pkl'))
scaler        = joblib.load(os.path.join(BASE, 'scaler.pkl'))
feature_names = joblib.load(os.path.join(BASE, 'feature_names.pkl'))

WEATHER_COLS = ['Weather_Foggy', 'Weather_Rainy', 'Weather_Snowy', 'Weather_Windy']
TRAFFIC_COLS = ['Traffic_Level_Low', 'Traffic_Level_Medium']


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    try:
        distance = float(data['distance'])
        prep     = float(data['prep'])
        exp      = float(data['exp'])
        weather  = data['weather'] 
        traffic  = data['traffic']  
    except (KeyError, ValueError) as e:
        return jsonify({'error': f'Invalid input: {e}'}), 400

    row = {name: 0 for name in feature_names}
    row['Distance_km']           = distance
    row['Preparation_Time_min']  = prep
    row['Courier_Experience_yrs'] = exp

    weather_col = f'Weather_{weather}'
    if weather_col in row:
        row[weather_col] = 1

    traffic_col = f'Traffic_Level_{traffic}'
    if traffic_col in row:
        row[traffic_col] = 1

    X = np.array([[row[f] for f in feature_names]])
    X_scaled = scaler.transform(X)
    prediction = float(model.predict(X_scaled)[0])
    prediction = max(1.0, prediction)

    return jsonify({
        'prediction': round(prediction, 1),
        'mae': 5.92 
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
