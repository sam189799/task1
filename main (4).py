'''

Welcome to GDB Online.
GDB online is an online compiler and debugger tool for C, C++, Python, Java, PHP, Ruby, Perl,
C#, OCaml, VB, Swift, Pascal, Fortran, Haskell, Objective-C, Assembly, HTML, CSS, JS, SQLite, Prolog.
Code, Compile, Run and Debug online from anywhere in world.

'''
from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

app = Flask(_name_)
model = joblib.load("iris_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    sl = float(request.form['sepal_length'])
    sw = float(request.form['sepal_width'])
    pl = float(request.form['petal_length'])
    pw = float(request.form['petal_width'])
    
    pred = model.predict([[sl, sw, pl, pw]])[0]
    species = ['setosa', 'versicolor', 'virginica'][pred]
    return render_template("index.html", prediction_text=f"Predicted species: {species}")

# Optional: REST API endpoint
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)
    features = [data['sepal_length'], data['sepal_width'], data['petal_length'], data['petal_width']]
    pred = model.predict([features])[0]
    species = ['setosa', 'versicolor', 'virginica'][pred]
    return jsonify({"prediction": species})

if _name_ == "_main_":
    app.run(debug=True)
    
    from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

iris = load_iris()
X, y = iris.data, iris.target
model = RandomForestClassifier()
model.fit(X, y)
joblib.dump(model, "iris_model.pkl")


    
    
