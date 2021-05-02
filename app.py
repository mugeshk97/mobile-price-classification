from service.pred import predict
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['POST'])
def main():
    if request.method == 'POST':
        df = request.get_json()
        inp = list(df.values())
        inp_array = np.array(inp, dtype = 'int64')
        inp_array = inp_array.reshape(1,-1)
        pred = predict(inp_array)[0]
        result = {'prediction': str(pred)}
    return jsonify(result)

if __name__ == '__main__':
	app.run(debug=True)