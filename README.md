# Predicting Loan Approval Using H2O
This is an end-to-end data science project to create a loan approval prediction. This project aims to create a loan prediction model, deploy it using FastAPI, and dockerize it for future deployment.

## Installation
Clone this repo: https://github.com/nugroho1234/loan-classification-h20-xgboost.git
The libraries I used for this project are:
1. pandas==2.0.3
2. numpy==1.24.4
3. joblib==1.3.2
4. h2o==3.42.0.2
5. fastapi==0.103.1
6. uvicorn==0.23.2
7. pydantic==2.3.0
8. scikit-learn==1.3.0

## Project Motivation
My main motivation in creating this project, besides honing my general skills, is trying out H2O library to create a prediction model. 

The tasks involved in creating the prediction model are:
1. Performing exploratory data analysis for each variable
2. Examining the relationship between the predictor variables and the target variable
3. Clean the data since there are lots of wrong data
4. Impute the missing values since there are lots of them
5. Create three baseline models using H2O GBM, H2O deep learning, and XGBoost
6. Perform feature engineering and feature selection to enhance the base model results
7. Experiment with the features generated and create new models to see whether the result is better or not
8. Pick the best model
9. Create a prediction pipeline
10. Deploy the model locally using FastAPI
11. Dockerize the app to be used in the future

## File Descriptions
The notebook files are as follows:
### 1-Single Variable EDA
This file describes the steps I took to perform EDA for each variable.
### 2-Relationship between variables and data imputation and preprocessing
This file describes my steps to see the relationship between the predictor and target variables. Here, I created a lot of plots to examine the relationship between variables. Then, I performed data imputation to fill in the missing values. Next, I did the data preprocessing steps to prepare for the model training phase. 
### 3-Model training
In this notebook, I trained the base models and examined their performances.
### 4-Feature engineering and selection
Here, I created a couple of new features to test whether using these features during training increased the model prediction scores. Then, I select the best model out of the base models and the model generated from the first and second experiments.
### 5-Model prediction testing
I created a prediction pipeline in the last notebook to preprocess and predict new data. Then, after creating an app using FastAPI, I checked the prediction again.
### helper.py
This file consists of helper functions I created while performing the EDA, feature engineering, and model training.
### app/main.py
This is the main file to run the API.
### app/loan_classification_variables.py
In this file, I used pydantic to predetermine the datatypes, which will be pushed to the API to predict the new data.
### app/preprocess.py
This file consists of helper functions to preprocess new data and create predictions using the selected model.
### app/Dockerfile
This file contains the dockerization process of the app.
### app/requirements.txt
The file consists of the libraries I used for this project.
### the data folder
The folder contains the dataset I used in the project and a few csv files I saved while doing the project.
### the model folder
The folder contains the models generated from this project and the dictionaries I used to create the prediction pipeline.

# How to Use This Project
You can examine my notebook and perform similar or different EDA and preprocessing steps. 

If you're interested in running the API, the command to run it is:
uvicorn main:LoanPredApp --reload

If you want to build the docker container, make sure you've downloaded docker for Windows. Use this command:
docker build -t "name of your project" 
to build the docker image. Then, use this command:
docker run -p 5000:5000 loan-prediction-app
to run the docker image.

Don't forget to download and install java to ensure the H2O server is running.

# Licensing, Author, Acknowledgements
I made this project to hone my skills and try the H2O library. If you can improve this, I will gladly hear from you! Feel free to use/tweak this and mention it to me so I can look at your work.

As usual, I benefit greatly from Stackoverflow and ChatGPT. I won't be able to live without them :)
