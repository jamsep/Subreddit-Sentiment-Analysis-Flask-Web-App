import json
from flask import Flask, request, redirect, render_template, session, url_for
from static import Subreddit_Sentiment_Analysis_Report as ssar
import pandas as pd
import numpy as np
import sys


app = Flask(__name__)
app.secret_key = 'secret-omg'

report_list = []
subreddit_name = ''
ID = ''
SECRET = ''



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form.get('user_input')
        if user_input == '':
            return redirect('/')
        subreddit = user_input
        session['subreddit'] = subreddit
        global subreddit_name
        subreddit_name = subreddit
        return redirect(url_for('processing', user_input=subreddit))

    subreddit = session.get('subreddit', None)
    session['subreddit'] = None
    return render_template('index.html')

@app.route('/processing/<user_input>', methods=['GET', 'POST'])
def processing(user_input):

    if request.method == "GET":
        subreddit_report = ssar.doSentimentReport(user_input.lower())
        global report_list
        if subreddit_report == None:
            return redirect(url_for('error'))
        elif type(subreddit_report) is tuple:
            print("BAD AUTHENTICATION", file=sys.stdout)
            global ID
            global SECRET
            ID = subreddit_report[0]
            SECRET = subreddit_report[1]
            return redirect(url_for('authenticationFailed'))
        else:
            report_list = subreddit_report
            return redirect(url_for('report'))
        

@app.route('/report/')
def report():
    split_data = clean_data()

    df_fdist_values = list(split_data[0].keys())
    df_fdist_data = np.array(list(split_data[0].values())).tolist() # convert type int64 to int in order for json parser to work
    sentiment_data = np.array(list(split_data[1].values())).tolist() # convert type int64 to int in order for json parser to work
    sentiment_values = list(split_data[1].keys())

    print(df_fdist_data, file=sys.stdout)
    print(df_fdist_values, file=sys.stdout)
    print(sentiment_data, file=sys.stdout)
    print(sentiment_values, file=sys.stdout)

    return render_template('report.html', name=subreddit_name, fdist_data = df_fdist_data, fdist_labels = df_fdist_values, sentiment_data = sentiment_data, sentiment_labels = sentiment_values)


@app.route('/error/')
def error():
    return render_template('error.html', subreddit_name=subreddit_name), {"Refresh": "5; url=/"}

@app.route('/authentication-failed/')
def authenticationFailed():
    return render_template('failed.html', ID=ID, SECRET=SECRET), {"Refresh": "5; url=/"}


def clean_data():
    split_data = []

    print(report_list, file=sys.stdout)
    df_fdist = report_list[0]
    df_cleaned = report_list[1]

    counts = df_cleaned.label.value_counts(normalize=True) * 100.0
    df_counts = counts.to_frame()

    df_counts = counts.to_frame()
    count_pos = 0
    count_neg = 0
    count_neu = 0


    for i in range(len(df_counts.index)):
        name = df_counts.iloc[i-1].name
        if name == 1:
            count_pos = df_counts.iloc[i-1].label
        if name == 0:
            count_neu = df_counts.iloc[i-1].label
        if name == -1:
            count_neg = df_counts.iloc[i-1].label

    df_cleaned_labels = []

    for label in counts.index:
        if label == 0:
            label = 'Neutral'
        if label == 1:
            label = 'Positive'
        if label == -1:
            label = 'Negative'
        df_cleaned_labels.append(label)

    df_cleaned_dict = {df_cleaned_labels[df_cleaned_labels.index("Positive")]: count_pos, df_cleaned_labels[df_cleaned_labels.index("Negative")]: count_neg, df_cleaned_labels[df_cleaned_labels.index("Neutral")]: count_neu}
    
    df_fdist.columns = ['Frequency']
    df_fdist = df_fdist.nlargest(10, 'Frequency')
    df_fdist.index

    df_fdist_dict = {}

    for i in df_fdist.index:
        df_fdist_dict[i] = df_fdist['Frequency'][i]

    
    split_data.append(df_fdist_dict)
    split_data.append(df_cleaned_dict)

    return split_data

if __name__ == '__main__':
    app.run(debug=False)