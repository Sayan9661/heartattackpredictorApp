from flask import render_template, Flask, request
import numpy as np
from joblib import dump, load

app = Flask(__name__, template_folder='templates',
            static_folder='static')


@app.route("/", methods=['POST', 'GET'])
def hello_world():
    req_type = request.method
    if req_type == 'GET':
        return render_template("index.html", output="No output yet Please enter values")

    else:
        sex = float(request.form['sex'])
        cp = float(request.form['cp'])
        trtbps = float(request.form['trtbps'])
        chol = float(request.form['chol'])
        fbs = float(request.form['fbs'])
        restecg = float(request.form['restecg'])
        thalachh = float(request.form['thalachh'])
        exng = float(request.form['exng'])
        oldpeak = float(request.form['oldpeak'])
        slp = float(request.form['slp'])
        caa = float(request.form['caa'])
        thall = float(request.form['thall'])
        data = np.array([[sex, cp, trtbps, chol, fbs, restecg,
                         thalachh, exng, oldpeak, slp, caa, thall]])

        model = load('saved_model.joblib')

        preds = model.predict(data)

        return render_template("index.html", output=str(preds))
