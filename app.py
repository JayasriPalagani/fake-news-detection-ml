from flask import Flask, render_template, request
import pickle
import re

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

suspicious_words = [
    "shocking", "breaking", "secret", "miracle", "urgent",
    "unbelievable", "overnight", "doctors hate", "hidden",
    "leaked", "must read", "viral", "exposed"
]

def explain_news(text):
    text_lower = text.lower()

    found_words = []
    for word in suspicious_words:
        if word in text_lower:
            found_words.append(word)

    words = re.findall(r'\b[a-zA-Z]{4,}\b', text_lower)
    common_words = list(set(words))[:8]

    explanation = []

    if found_words:
        explanation.append("Suspicious/clickbait words found: " + ", ".join(found_words))

    explanation.append("Important words detected: " + ", ".join(common_words))

    return explanation

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["news"]

    vector = vectorizer.transform([text])
    result = model.predict(vector)
    prob = model.predict_proba(vector)[0]

    confidence = round(max(prob) * 100, 2)
    label = "REAL NEWS ✅" if result[0] == 1 else "FAKE NEWS ❌"
    explanation = explain_news(text)

    return render_template(
        "index.html",
        prediction=label,
        confidence=confidence,
        explanation=explanation
    )

if __name__ == "__main__":
    app.run(debug=True)