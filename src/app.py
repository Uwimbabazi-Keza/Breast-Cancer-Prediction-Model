from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('models/breast_cancer_model.keras')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [
        float(request.form['radius_mean']),
        float(request.form['texture_mean']),
        float(request.form['perimeter_mean']),
        float(request.form['area_mean']),
        float(request.form['smoothness_mean']),
        float(request.form['compactness_mean']),
        float(request.form['concavity_mean']),
        float(request.form['concave points_mean']),
        float(request.form['symmetry_mean']),
        float(request.form['fractal_dimension_mean']),
        float(request.form['radius_se']),
        float(request.form['texture_se']),
        float(request.form['perimeter_se']),
        float(request.form['area_se']),
        float(request.form['smoothness_se']),
        float(request.form['compactness_se']),
        float(request.form['concavity_se']),
        float(request.form['concave points_se']),
        float(request.form['symmetry_se']),
        float(request.form['fractal_dimension_se']),
        float(request.form['radius_worst']),
        float(request.form['texture_worst']),
        float(request.form['perimeter_worst']),
        float(request.form['area_worst']),
        float(request.form['smoothness_worst']),
        float(request.form['compactness_worst']),
        float(request.form['concavity_worst']),
        float(request.form['concave points_worst']),
        float(request.form['symmetry_worst']),
        float(request.form['fractal_dimension_worst'])
    ]

    features = np.array([features])
    prediction = model.predict(features)
    result = "Malignant" if prediction > 0.5 else "Benign"

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
