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

