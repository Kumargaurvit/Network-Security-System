from networksecurity.exception.exception import NetworkSecuirtyException
from networksecurity.logging.logger import logging
from networksecurity.constant.training_pipeline import SAVED_MODEL_DIR, MODEL_FILE_NAME

import os
import sys

class NetworkModel:
    def __init__(self, preprocessor, model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise NetworkSecuirtyException(e,sys)
    
    def predict(self,x):
        try:
            x_transformed = self.preprocessor.transform(x)
            y_hat = self.model.predict(x_transformed)
            return y_hat
        except Exception as e:
            raise NetworkSecuirtyException(e,sys)