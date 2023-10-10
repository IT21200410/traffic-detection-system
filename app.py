from flask import Flask, request, url_for, redirect, render_template
import pandas as pd
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app, support_credentials=True)

model = pickle.load(open("example_weights_knn.pkl", "rb"))
@app.route('/')
def use_template():
    return render_template('index.html')

@app.route('/predict', methods = ['POST', 'GET'])
def predict():
    input_one = request.form['1']
    input_two = request.form['2']
    input_three = request.form['3']
    input_four = request.form['4']
    input_five = request.form['5']
    input_six = request.form['6']
    input_seven = request.form['7']
    input_eight = request.form['8']

    setup_df = pd.DataFrame([pd.Series([input_one, input_two, input_three, input_four, input_five, input_six, input_seven, input_eight])])
    diabetas_prediction = model.predict_proba(setup_df)
    output = '{0:.1f}'.format(diabetas_prediction[0][1], 2)
    output = str(float(output)*100) + '%'

    if output > str(0.5):
        return render_template('result.html', pred = f'You have the followin chance of having diabetes based on our prediction')
    else:
        return render_template('result.html', pred = f'You have a low chance of diabetes which is currently considered')
    


