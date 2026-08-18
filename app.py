
from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load model
with open('rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load feature names
with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html', features=feature_names)

@app.route('/predict', methods=['POST'])
def predict():

    values = []

    for feature in feature_names:
        values.append(float(request.form[feature]))

    input_df = pd.DataFrame([values], columns=feature_names)

    prediction = model.predict(input_df)[0]

    return render_template(
        'index.html',
        features=feature_names,
        prediction=round(prediction, 2)
    )

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
