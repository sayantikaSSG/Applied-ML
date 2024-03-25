import joblib
from typing import Tuple

def score(text: str, model, threshold=0.5) -> Tuple[bool, float]:
    try:
        # Get the propensity score for the positive class
        propensity = model.decision_function([text])[0]
        prediction = (propensity >= threshold)
        
        return bool(prediction), propensity
    except Exception as e:
        print("Error scoring text:", e)
        return False, 0.0

# Load the model
try:
    # Adjust the path to where your model is stored
    best_model = joblib.load("/home/sayantika/Desktop/DS2sem4/Applied Machine Leaning/Applied-ML/assignment 3/best_model.joblib")

except Exception as e:
    print("Error loading model file:", e)
