from flask import Flask, render_template, request
import joblib

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
reverse_map = {0: "Low", 1: "Medium", 2: "High"}

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    title = request.form['title']
    abstract = request.form['abstract']
    text = title + " " + abstract
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)
    result = reverse_map[prediction[0]]
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
