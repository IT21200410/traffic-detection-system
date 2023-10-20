from flask import Flask, request, url_for, redirect, render_template
import pandas as pd
import pickle
import datetime
import calendar
from flask_cors import CORS


app = Flask(__name__)
CORS(app, support_credentials=True)


model = pickle.load(open("Randommodel.pkl", "rb"))
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/add', methods = ['POST'], strict_slashes = False)
def add_articles():
    title = request.json['title']
    body = request.json['body']

    return title


@app.route('/predict_manuja', methods = ['POST'], strict_slashes = False)
def predict_manuja():
    Junction = request.json['junction']
    date = request.json['date']
    # date = datetime.datetime.strptime(date,'%Y-%m-%d')
    time = request.json['time']
    df = preProcess(Junction, date, time)
    traffic_level = model.predict(df)
    return {traffic_level[0]}


def preProcess(junction, date, time):
    inputs = {'Junction': [junction], 'date': [date], 'time': [time]}       
    df = pd.DataFrame(inputs)
    df['date'] = pd.to_datetime(df['date'])
    df['week'] = df['date'].apply(week_of_month)
    df['day'] = df['date'].dt.day_name()
    df['traffic_time'] = df['time'].apply(lambda x: datetime.datetime.strptime(x, '%H:%M').strftime('%H:%M'))
    df['traffic_time'] = int(str(df['time'])[:2])                                           
    df['month'] = df['date'].dt.month
    df['day'] = df['day'].map({'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7})
    # for i in range(len(df)):
    #     df['traffic_time'][i] = int(str(df['traffic_time'][i])[:2])
    # df['traffic_time'] = df['traffic_time'].str.split(':')[0]
    # df['traffic_time'] = int(df['traffic_time'])
    df.drop('date', axis = 1, inplace=True)
    reorder = ['Junction','week','day','traffic_time','month']
    df = df[reorder]
    return df

def week_of_month(tgtdate):

    days_this_month = calendar.mdays[tgtdate.month]
    for i in range(1, days_this_month):
        d = datetime.datetime(tgtdate.year, tgtdate.month, i)
        if d.day - d.weekday() > 0:
            startdate = d
            break
    # now we canuse the modulo 7 appraoch
    return (tgtdate - startdate).days //7 + 1



def convertTo24H(time):
    try:
        time_obj = datetime.datetime.strptime(time, '%I:%M%p')
        time24_hr = time_obj.strftime('%H:%M')
        print(f'24 hr time is : {time24_hr}')
        return time24_hr
    except ValueError:
        return 'Invalid Time Format'



@app.route('/hansi')
def use_hansi():
    return render_template('traffic.html')

@app.route('/KNN')
def use_knn():

    KNNmodel = pickle.load(open("KNN_model.pkl", "rb"))

    input = [[1,3,3,11,2]]
    prediction = KNNmodel.predict(input)

    return render_template('KNN.html', prediction_text = "The Traffic level is {}".format(prediction))

    
if __name__ == '__main__':
    app.run(debug = True)

