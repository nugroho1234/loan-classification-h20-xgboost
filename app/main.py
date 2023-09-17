#fastapi
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
from loan_classification_variables import LoanPred, ThresholdMetrics

#h2o
import h2o

#standard library
import pandas as pd
import joblib
import os
import re

import preprocess

LoanPredApp = FastAPI()

#get model path
current_dir = os.getcwd()
model_filename = 'model/dl_grid_model_66'
knn_initial_filename = 'model/knn_imputer_model.pkl'
knn_cur_filename = 'model/knn_imputer_model_no_multicol.pkl'
scaler_filename = 'model/scaler_no_multicol.pkl'
purpose_filename = 'model/purpose_mapping.pkl'

model_path = os.path.join(current_dir, model_filename)
knn_initial_path = os.path.join(current_dir, knn_initial_filename)
knn_cur_path = os.path.join(current_dir, knn_cur_filename)
scaler_path = os.path.join(current_dir, scaler_filename)
purpose_path = os.path.join(current_dir, purpose_filename)

#initialize h2o
h2o.init()

model = h2o.load_model(model_path)
knn_initial_model = joblib.load(knn_initial_path)
knn_cur_model = joblib.load(knn_cur_path)
scaler = joblib.load(scaler_path)
purpose_mapping = joblib.load(purpose_path)

@LoanPredApp.get('/')
def index():
    #default return message
    return{'message': 'use /predict to make loan prediction'}

@LoanPredApp.post('/predict')
def predict_price(data: LoanPred, threshold: ThresholdMetrics):
    #convert JSON payload to dictionary
    data = data.dict()
    
    #use preprocess function to create dataframe with features matching the training and validation dataset
    df = preprocess.create_dataframe(data, knn_initial_model, purpose_mapping, knn_cur_model, scaler)
    
    #create h2o dataframe
    hf = h2o.H2OFrame(df)
    
    #predict the data 
    threshold_metrics = threshold.threshold_metrics
    prediction = preprocess.predict_data(model, hf, threshold_metrics=threshold_metrics)
        
    return JSONResponse(content=prediction)

if __name__ == '__main__':
    uvicorn.run("main:LoanPredApp",host='0.0.0.0', port=5000, reload=True, workers=4)

