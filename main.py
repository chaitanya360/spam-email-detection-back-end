import json
from urllib import response
from flask import Flask, render_template, request, jsonify
import pickle

# init Flask App

STATIC_FOLDER = 'template/assets'


app = Flask(__name__, template_folder='template', static_folder=STATIC_FOLDER)


vect = pickle.load(open('vector', 'rb'))

# Load Pickle models
DT_loaded_model = pickle.load(open('DT', 'rb'))
NB_loaded_model = pickle.load(open('NB', 'rb'))
SVM_loaded_model = pickle.load(open('SVM', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


def parseData(data):
    booleans = [False, True]
    return booleans[(data[0])]


@app.route('/predict/<string:message>', methods=['GET'])
def predict(message):
    vectorized_input_data = vect.transform([message])

    dt_prediction = DT_loaded_model.predict(vectorized_input_data)
    nb_prediction = NB_loaded_model.predict(vectorized_input_data)
    svm_prediction = SVM_loaded_model.predict(vectorized_input_data)

    response = jsonify({"dt_pred": parseData(dt_prediction), "nb_pred": parseData(
        nb_prediction), "svm_pred": parseData(svm_prediction)})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


if __name__ == '__main__':
    app.run(debug=True)
