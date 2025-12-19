from networksecurity.exception.exception import NetworkSecuirtyException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact

from networksecurity.utils.main_utils.utils import save_object, load_object
from networksecurity.utils.main_utils.utils import load_numpy_array_data, evaluate_models
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

import os
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)
import mlflow
import dagshub
from urllib.parse import urlparse
# dagshub.init(repo_owner='Kumargaurvit', repo_name='Network-Security', mlflow=True)

class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                model_trainer_config: ModelTrainerConfig):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
        except Exception as e:
            raise NetworkSecuirtyException(e,sys)
    
    def setup_mlflow(self):
        """
        Configure MLflow for DagsHub using token-based auth.
        This MUST NOT run in production containers.
        """
        os.environ["MLFLOW_TRACKING_URI"] = (
            "https://dagshub.com/Kumargaurvit/Network-Security.mlflow"
        )
        os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
        os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")
        
    def track_mlflow(self,best_model,classification_metric):
        try:
            if os.getenv("ENV") == "production":
                return
            
            self.setup_mlflow()

            mlflow.set_registry_uri("https://dagshub.com/Kumargaurvit/Network-Security.mlflow")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            with mlflow.start_run():
                mlflow.log_metric("f1_score", classification_metric.f1_score)
                mlflow.log_metric("precision_score", classification_metric.precision_score)
                mlflow.log_metric("recall_score", classification_metric.recall_score)

                mlflow.sklearn.log_model(best_model,name="model")

        except Exception as e:
            raise NetworkSecuirtyException(e, sys)
        
    def train_model(self,X_train,y_train,X_test,y_test):
        try:
            # Initializing the models
            models = {
                "Logistic Regression" : LogisticRegression(verbose=1),
                "Decision Tree Classifier" : DecisionTreeClassifier(),
                "Ada Boost Classifier" : AdaBoostClassifier(),
                "Gradient Boosting Classifier" : GradientBoostingClassifier(verbose=1),
                "Random Forest Classifier" : RandomForestClassifier(verbose=1),
            }

            # Defining Hyperparamters
            params = {
                "Logistic Regression" : {},

                "Decision Tree Classifier" : {
                    "criterion" : ['gini','entropy','log_loss'],
                    "splitter" : ['best','random'],
                    "max_features" : ['sqrt','log2']
                },

                "Ada Boost Classifier" : {
                    "learning_rate" : [0.1,0.01,0.05,0.001],
                    "n_estimators" : [8,16,32,64,128,256]
                },

                "Gradient Boosting Classifier" : {
                    "loss" : ['log_loss','exponential'],
                    "learning_rate" : [0.1,0.01,0.05,0.001],
                    "criterion" : ['squared_error','friedman_mse'],
                    "max_features" : ['auto','sqrt','log2'],
                    "n_estimators" : [8,16,32,64,128,256]  
                },
                
                "Random Forest Classifier" : {
                    "criterion" : ['gini','entropy','log_loss'],
                    "n_estimators" : [8,16,32,64,128,256],
                    "max_features" : ['sqrt','log2',None]
                },
            }

            logging.info('Evaluating the Models')
            model_report: dict = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)
            
            logging.info('Selecting the best Model')
            # Retrieving Best Model Score and Name
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            logging.info(f"Best Model : {best_model_name} Selected")
            best_model = models[best_model_name]
            y_train_pred = best_model.predict(X_train)

            classification_train_metric = get_classification_score(y_train,y_train_pred)
            
            # Tracking the Experiments with MLFlow
            # self.track_mlflow(best_model,classification_train_metric)

            y_test_pred = best_model.predict(X_test)
            classification_test_metric = get_classification_score(y_test,y_test_pred)

            # Tracking the Experiments with MLFlow
            # self.track_mlflow(best_model,classification_test_metric)

            preprocessor = load_object(self.data_transformation_artifact.transformed_object_file_path)
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path,exist_ok=True)

            logging.info('Saving the Model')
            Network_model = NetworkModel(preprocessor,best_model)
            save_object(self.model_trainer_config.trained_model_file_path, obj=Network_model)
            save_object('final_models/model.pkl',best_model)

            logging.info('Creating Model Trainer Artifacts')
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric
            )

            return model_trainer_artifact
        
        except Exception as e:
            raise NetworkSecuirtyException(e,sys)
    
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info('Model Trainer Started')
            train_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)

            # Train and Test Splits
            logging.info('Creating Train and Test Splits')
            X_train, y_train = (
                train_arr[:,:-1],
                train_arr[:,-1]
            )
            X_test, y_test = (
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            logging.info('Training the Models')
            model_trainer_artifact = self.train_model(X_train,y_train,X_test,y_test)
            return model_trainer_artifact
        
        except Exception as e:
            raise NetworkSecuirtyException(e,sys)