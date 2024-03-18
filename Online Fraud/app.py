pip install Flask
from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('path_to_your_trained_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the form
    step = int(request.form['step'])
    type_ = request.form['type']
    amount = int(request.form['amount'])
    nameOrig = request.form['nameOrig']
    oldbalanceOrg = int(request.form['oldbalanceOrg'])
    newbalanceOrig = int(request.form['newbalanceOrig'])
    nameDest = request.form['nameDest']
    oldbalanceDest = int(request.form['oldbalanceDest'])
    newbalanceDest = int(request.form['newbalanceDest'])

    # Create a DataFrame from the input data
    data = pd.DataFrame({
        'step': [step],
        'type': [type_],
        'amount': [amount],
        'nameOrig': [nameOrig],
        'oldbalanceOrg': [oldbalanceOrg],
        'newbalanceOrig': [newbalanceOrig],
        'nameDest': [nameDest],
        'oldbalanceDest': [oldbalanceDest],
        'newbalanceDest': [newbalanceDest]
    })

    # Make a prediction
    prediction = model.predict(data)[0]

    # Return the prediction result
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
