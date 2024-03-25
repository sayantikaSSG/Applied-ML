import unittest
import joblib
from score import score  # Adjust this import to the actual location of your score function

class TestScoreFunction(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Load the model once for all tests
        try:
            cls.model = joblib.load("/home/sayantika/Desktop/DS2sem4/Applied Machine Leaning/Applied-ML/assignment 3/best_model.joblib")
        except Exception as e:
            raise Exception("Model loading failed: {}".format(e))

    def test_smoke(self):
        """Smoke test to ensure the function produces some output without crashing."""
        _, propensity = score("test", self.model)
        self.assertIsNotNone(propensity)
    
    def test_io_format(self):
        """Test if the input/output formats/types are as expected."""
        result, propensity = score("test", self.model)
        self.assertIsInstance(result, bool)
        self.assertIsInstance(propensity, float)
    
    def test_prediction_value(self):
        """Test if the prediction value is 0 or 1."""
        result, _ = score("test", self.model)
        self.assertIn(result, [True, False])
    
    def test_propensity_score_range(self):
        """Test if the propensity score is between 0 and 1."""
        _, propensity = score("test", self.model, threshold=0.5)
        self.assertGreaterEqual(propensity, 0)
        self.assertLessEqual(propensity, 1)
    
    def test_threshold_zero(self):
        """Test if setting the threshold to 0 makes the prediction always 1 (True)."""
        result, _ = score("test", self.model, threshold=0)
        self.assertTrue(result)
    
    def test_threshold_one(self):
        """Test if setting the threshold to 1 makes the prediction always 0 (False)."""
        result, _ = score("test", self.model, threshold=1)
        self.assertFalse(result)
    
    def test_obvious_spam(self):
        """Test with an obvious spam input text that the prediction is 1."""
        spam_text = "FREE lottery tickets!!"
        result, _ = score(spam_text, self.model)
        self.assertTrue(result)
    
    def test_obvious_non_spam(self):
        """Test with an obvious non-spam input text that the prediction is 0."""
        non_spam_text = "Hello, how are you?"
        result, _ = score(non_spam_text, self.model)
        self.assertFalse(result)

if __name__ == "__main__":
    unittest.main()
import unittest
import requests
from app import app  # Import the Flask app

class TestFlaskApp(unittest.TestCase):

    def setUp(self):
        # Setup Flask test client
        app.testing = True
        self.client = app.test_client()

    def test_flask_endpoint(self):
        # Send a POST request to the /score endpoint
        response = self.client.post('/score', json={'text': 'FREE lottery tickets!!'})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('prediction', data)
        self.assertIn('propensity', data)

if __name__ == '__main__':
    unittest.main()
