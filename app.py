from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# Load data
data = pd.read_csv("D:\\Language Detection using ML\\dataset.csv")

# Preprocessing
data.isnull().sum()
data["language"].value_counts()
x = np.array(data["Text"])
y = np.array(data["language"])

# Feature extraction
cv = CountVectorizer()
X = cv.fit_transform(x)
# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.33,
                                                    random_state=42)

# Model training
model = MultinomialNB()
model.fit(X_train, y_train)

# Function to predict language
def predict_language(text):
    data = cv.transform([text]).toarray()
    output = model.predict(data)
    return output[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    prediction = predict_language(text)
    return prediction

if __name__ == '__main__':
    app.run(debug=True)