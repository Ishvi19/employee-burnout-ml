from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("employee_burnout_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    gender = int(request.form["gender"])
    company_type = int(request.form["company_type"])
    wfh = int(request.form["wfh"])
    designation = int(request.form["designation"])
    resource = int(request.form["resource"])
    fatigue = float(request.form["fatigue"])

    data = np.array([[gender, company_type, wfh, 
                      designation, resource, fatigue]])

    prediction = model.predict(data)[0]

    return render_template("index.html",
                           result=f"Burnout Score: {round(prediction,2)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)