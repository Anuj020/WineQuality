import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import os

class DataTransformationConfig:
    preprocesser_obj_file_path  = os.path.join('artifacts',"preprocesser.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):

        '''
                This function is responsible for Data transformation
        '''

        try:
            numeric_columns = [
                "fixed acidity","volatile acidity","citric acid","residual sugar","chlorides",
                "free sulfur dioxide","total sulfur dioxide","density","pH","sulphates",
                "alcohol"
            ] 
            
            num_pipeline = Pipeline(
                steps= [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]   
            )

            logging.info(f"Numeric columns:{numeric_columns}")

            preprocesser = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numeric_columns)
                ]
            )

            return preprocesser
        
        except Exception as e:
            raise CustomException(e,sys)
    

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocesser_obj = self.get_data_transformer_object()

            target_column_name = "quality"
            numeric_columns = [
                "fixed acidity","volatile acidity","citric acid","residual sugar","chlorides",
                "free sulfur dioxide","total sulfur dioxide","density","pH","sulphates",
                "alcohol"
            ]

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]


            logging.info("Applying preprocessing object on training dataframe and  testing dataframe.")

            input_feature_train_arr = preprocesser_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocesser_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
 
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info(f"saved preprocessing object. ")

            save_object(
                file_path = self.data_transformation_config.preprocesser_obj_file_path,
                obj = preprocesser_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocesser_obj_file_path
            )



        except Exception as e:
            raise CustomException(e,sys)