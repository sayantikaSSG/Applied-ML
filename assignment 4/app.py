from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Attempt to load the trained model and vectorizer
try:
    loaded_model = pickle.load(open("/home/sayantika/Desktop/DS2sem4/Applied Machine Leaning/Applied-ML/assignment 4/Models/logistic_regression_best_model.pkl", 'rb'))
except Exception as e:
    print(f"Failed to load the model: {e}")
    raise e

@app.route('/score', methods=['POST'])
def score_endpoint():
    data = request.get_json()
    text = data.get('text')
    if text:
        from score import score
        prediction, propensity = score(text, loaded_model, threshold=0.5)
        return jsonify({'prediction': prediction, 'propensity': propensity})
    else:
        return jsonify({'error': 'No text provided'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
