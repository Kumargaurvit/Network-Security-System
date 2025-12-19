import sys
import os
import pymongo
import pandas as pd

import certifi
ca = certifi.where()

from dotenv import load_dotenv
load_dotenv()

mongo_db_url = os.getenv("MONGO_DB_URL")

from networksecurity.exception.exception import NetworkSecuirtyException
from networksecurity.logging.logger import logging
from networksecurity.pipeline.training_pipeline import TrainingPipeline
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.constant.training_pipeline import DATA_INGESTION_COLLECTION_NAME, DATA_INGESTION_DATABASE_NAME
from networksecurity.constant.training_pipeline import PHISHING, LEGITIMATE

from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.feature_extractor.extractor import extract_features_from_url

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request, Form
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse

client = pymongo.MongoClient(mongo_db_url, tlsCAFile = ca)

database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory='./templates')

@app.get('/', tags=['authentication'])
async def index():
    return RedirectResponse(url='/docs')

@app.get('/train')
async def train_route():
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response('Model Training is Successful')
    except Exception as e:
        raise NetworkSecuirtyException(e,sys)
    
@app.post('/predict')
async def predict_route(request:Request,url: str = Form(None,description="Enter URL to Detect")):
    try:
        features = extract_features_from_url(url)
        print("Extracted Features:", features)

        columns = [
            'having_IP_Address','URL_Length','Shortining_Service',
            'having_At_Symbol','double_slash_redirecting','Prefix_Suffix',
            'having_Sub_Domain','SSLfinal_State','Domain_registeration_length',
            'Favicon','port','HTTPS_token','Request_URL','URL_of_Anchor',
            'Links_in_tags','SFH','Submitting_to_email','Abnormal_URL',
            'Redirect','on_mouseover','RightClick','popUpWidnow','Iframe',
            'age_of_domain','DNSRecord'
        ]

        df = pd.DataFrame([features], columns=columns)
        print(df)

        preprocessor = load_object('final_models/preprocessor.pkl')
        final_model = load_object('final_models/model.pkl')
        network_model = NetworkModel(preprocessor,final_model)

        y_pred = network_model.predict(df)
        print("Raw prediction:", y_pred)

        if int(y_pred[0]) == PHISHING:
            prediction = "Phishing"
        if int(y_pred[0]) == LEGITIMATE:
            prediction = "Legitimate"
        
        print("Mapped prediction:", prediction)

        df['predicted_column'] = y_pred

        table_html = df.to_html(classes='table table-striped')

        return templates.TemplateResponse("table.html",{"request": request,"table": table_html,"url": url,"prediction": prediction})
    except Exception as e:
        raise NetworkSecuirtyException(e,sys)
    
if __name__ == "__main__":
    app_run(app,host="0.0.0.0",port=8000)