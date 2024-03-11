import pytest
from score import score
import joblib

# Load the model for testing
model = joblib.load('assignment 3/spam_classifier.pkl')

def test_score():
    # Smoke test
    prediction, propensity = score("Free money!!!", model, 0.5)
    assert isinstance(prediction, bool)
    assert isinstance(propensity, float)
    
    # Format test
    assert 0 <= propensity <= 1
    
    # Threshold tests
    assert score("Free money!!!", model, 0.0)[0] == True
    assert score("Hello, how are you?", model, 1.0)[0] == False
    
    # Prediction based on obvious texts
    assert score("This is definitely not spam.", model, 0.5)[0] == False
    assert score("Buy now, last chance!!!", model, 0.5)[0] == True
