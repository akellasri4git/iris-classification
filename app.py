from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict

app = FastAPI()



class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.post("/predict")
def predict_flower(request: IrisRequest):

    sample = [[
        request.sepal_length,
        request.sepal_width,
        request.petal_length,
        request.petal_width
    ]]

    result = predict(sample)

    return {
        "prediction": result
    }