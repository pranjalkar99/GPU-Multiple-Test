from flask import Flask
import requests
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/run_model/<int:model_number>', methods=['GET'])
def run_model(model_number):
    if model_number == 1:
        try:
            response = requests.get('http://model1:5001/predict')
            print(response)
        except requests.exceptions.ConnectionError:
            return "gpu 1 unavailable"
    elif model_number == 2:
        try:
            response = requests.get('http://model1:5002/predict_mobilenet')
            print(response)
        except requests.exceptions.ConnectionError:
            return "gpu 2 unavailable"
    else:
        return "Invalid model number"
    return response.text
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)
