from flask import Flask,jsonify,request,make_response,url_for,redirect
import requests, json
from loading import predict
app = Flask(__name__)


@app.route('/predict', methods=['GET','POST'])
def create_row_in_gs():
    if request.method == 'GET':
        return make_response('failure')
    if request.method == 'POST':
        img = request.json['id']

        response = requests.post(
            data=json.dumps(predict(img)),
            headers={'Content-Type': 'application/json'}
        )
        return response.content

if __name__ == '__main__':
    app.run(host='localhost',debug=False, use_reloader=True)