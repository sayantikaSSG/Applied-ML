from flask import Flask, request, jsonify
import joblib
from score import score  # Adjust this import based on your project structure

app = Flask(__name__)

# Load the model (consider doing this in a more efficient way for production)
model = joblib.load("best_model.joblib")

@app.route('/score', methods=['POST'])
def score_text():
    if request.method == 'POST':
        data = request.get_json()
        text = data.get("text", "")
        if text:
            prediction, propensity = score(text, model)
            return jsonify({"prediction": prediction, "propensity": propensity})
        else:
            return jsonify({"error": "No text provided"}), 400

if __name__ == '__main__':
    app.run(debug=True)
