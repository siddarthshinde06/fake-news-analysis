from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load model & vectorizer
vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("lr_model.jb")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["news"]

    transformed = vectorizer.transform([text])
    prediction = model.predict(transformed)[0]

    result = "Real News" if prediction == 1 else "Fake News"
    color = "green" if prediction == 1 else "red"

    return render_template("index.html", prediction=result, color=color, news=text)

if __name__ == "__main__":
    app.run(debug=True)
