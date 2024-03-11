import joblib

def score(text: str, model, threshold: float) -> (bool, float):
    text_transformed = model.named_steps['tfidfvectorizer'].transform([text])
    propensity = model.named_steps['logisticregression'].predict_proba(text_transformed)[:, 1][0]
    prediction = propensity > threshold
    return prediction, propensity
