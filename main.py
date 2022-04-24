from flask import Flask, render_template, request
import pickle

# init Flask App

STATIC_FOLDER = 'template/assets'


app = Flask(__name__, template_folder='template', static_folder=STATIC_FOLDER)


tfvect = pickle.load(open('DT_vector', 'rb'))

# Load Pickle model
loaded_model = pickle.load(open('DT', 'rb'))


def fake_news_det(news):
    input_data = [news]
    vectorized_input_data = tfvect.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    return prediction

# Defining the site route


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        pred = fake_news_det(message)
        print(pred[0])
        return render_template('index.html', prediction=pred[0])
    else:
        return render_template('index.html', prediction="Something went wrong")


if __name__ == '__main__':
    app.run(debug=True)
