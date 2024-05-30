from flask import Flask, request, jsonify
import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer

app = Flask(__name__)

model = joblib.load('breast_cancer_model.pkl')

data = load_breast_cancer()
feature_names = data.feature_names

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json

        features = np.array([input_data.get(feat, 0) for feat in feature_names]).reshape(1, -1)

        prediction = model.predict(features)

        return jsonify({
            'prediction': int(prediction[0]),
            'class': 'Malignant' if prediction[0] else 'Benign'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/')
def home():
    return "Breast Cancer Prediction API"

if __name__ == '__main__':
    app.run(debug=True)
