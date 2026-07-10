import joblib
from src.config import MODEL_PATH

model = joblib.load(MODEL_PATH)

def predict(sample):

    prediction = model.predict(sample)

    return prediction[0]