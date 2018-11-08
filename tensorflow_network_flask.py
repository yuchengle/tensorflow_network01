# coding=utf-8
from flask import Flask
import tensorflow_network_restore as tfnr

app = Flask(__name__)
PORT=5001

@app.route('/')
def index2():
    file = 'test0.txt'
    return tfnr.getresult(file)

if __name__ == '__main__':
    app.run(debug=True, port=PORT)
