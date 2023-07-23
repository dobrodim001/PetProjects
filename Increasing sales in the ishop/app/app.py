import pickle
import xgboost as xgb

import fastapi
import numpy as np


app = fastapi.FastAPI(title="ML API", description="API for classification items", version="1.0")


@app.on_event('startup')
def load_model():
    global model        
    model = pickle.load(open('xgb_model.sav', 'rb'))


@app.post('/predict', tags=["predictions"])
async def get_prediction(data: list[list[float]]):
    global model
    data = np.array(data)
    prediction = model.predict(data).tolist()
    
    return {"prediction": prediction}
