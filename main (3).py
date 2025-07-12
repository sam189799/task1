'''

Welcome to GDB Online.
GDB online is an online compiler and debugger tool for C, C++, Python, Java, PHP, Ruby, Perl,
C#, OCaml, VB, Swift, Pascal, Fortran, Haskell, Objective-C, Assembly, HTML, CSS, JS, SQLite, Prolog.
Code, Compile, Run and Debug online from anywhere in world.

'''
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['species'] = df['target'].map(dict(enumerate(iris.target_names)))

# EDA visualization
sns.pairplot(df, hue="species")
plt.show()

print(df.describe())
print(df['species'].value_counts())import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['species'] = df['target'].map(dict(enumerate(iris.target_names)))

# EDA visualization
sns.pairplot(df, hue="species")
plt.show()

print(df.describe())

print(df['species'].value_counts())

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Feature and target separation
X = df[iris.feature_names]
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "iris_model.pkl")

from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(_name_)
model = joblib.load("iris_model.pkl")

@app.route("/")
def home():
    return "Iris Classification Model is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    features = [data['sepal_length'], data['sepal_width'], data['petal_length'], data['petal_width']]
    prediction = model.predict([features])[0]
    species = ['setosa', 'versicolor', 'virginica'][prediction]
    return jsonify({"prediction": species})

if _name_ == "_main_":
    app.run(debug=True)

curl -X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width":0.2}'


{"prediction": "setosa"}












