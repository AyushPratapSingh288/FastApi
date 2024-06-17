import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import sklearn
print(sklearn.__version__)
import pickle

app = FastAPI()


try:
    model = pickle.load(open('classifiernew.pkl', 'rb'))
    print("Model loaded successfully!")
except OSError as e:
    print(f"Error loading model: {e}")  # Likely file system error
except Exception as e:  # Catch other unexpected exceptions
    print(f"Unexpected error: {e}")


class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float


@app.get('/')
def index():
    return {'message': 'Hello, World'}


@app.post("/predict")
def predict_banknote(data: InputData):
    data = data.dict()
    variance = data['feature1']
    skewness = data['feature2']
    curtosis = data['feature3']
    entropy = data['feature4']

    prediction = model.predict([[variance, skewness, curtosis, entropy]])
    if prediction[0] > 0.5:
        prediction = "Fake note"
    else:
        prediction = "Its a Bank note"
    return {
        'prediction': prediction
    }


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
