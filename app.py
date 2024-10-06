from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from keras.models import load_model
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Load the trained model
model = load_model("./models/model3.keras")


#  InputData
#     ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity, Potability
class InputData(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float


# Initialize FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "Welcome to the Dementia Prediction API"}


@app.post("/predict")
def predict(data: InputData):

    # Make prediction
    prediction = model.predict(data)
    print(prediction[0])

    # Return the prediction as a response
    return {"prediction": round(prediction[0], 0)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
