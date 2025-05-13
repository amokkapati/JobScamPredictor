from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")  # Save it when you train

@app.route("/", methods=["GET", "POST"])
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None

    if request.method == "POST":
        input_text = request.form.get("jobtext", "")
        if input_text.strip():
            X_input = vectorizer.transform([input_text])
            probs = model.predict_proba(X_input)[0]
            pred_class = model.predict(X_input)[0]
            prediction = "Fake" if pred_class == 1 else "Real"
            confidence = round(probs[pred_class] * 100, 2)
    
    return render_template("index.html", prediction=prediction, confidence=confidence)


if __name__ == "__main__":
    app.run(debug=True)
