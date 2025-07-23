import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np
import os

from sklearn.compose import ColumnTransformer # to apply transformations to specific columns
from sklearn.impute import SimpleImputer # to handle missing values
from sklearn.pipeline import Pipeline # to create a pipeline of transformations
from sklearn.preprocessing import StandardScaler, OneHotEncoder # for scaling and encoding categorical variables

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        try:
            numerical_features = ['writing_score', 'reading_score']
            categorical_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']        
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore')),
                ('scaler', StandardScaler(with_mean=False))  # Scaling after one-hot encoding, avoids centering sparse matrix
            ])
            logging.info("Numerical and Categorical pipelines created")

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', num_pipeline, numerical_features),
                    ('cat', cat_pipeline, categorical_features)
                ]
            )
            logging.info("ColumnTransformer created with numerical and categorical pipelines")
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys) from e
    
    def initiate_data_transformation(self, train_path, test_path):
        logging.info("Data Transformation started")
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and Test data read successfully")

            preprocessor_obj = self.get_data_transformer_object()
            target_column_name = 'math_score'
            numerical_columns = ['writing_score', 'reading_score']

            X_train = train_df.drop(columns=[target_column_name], axis=1)
            y_train = train_df[target_column_name]

            X_test = test_df.drop(columns=[target_column_name], axis=1)
            y_test = test_df[target_column_name]
            logging.info("Features and target variable separated")

            X_train_transformed = preprocessor_obj.fit_transform(X_train)
            X_test_transformed = preprocessor_obj.transform(X_test)
            logging.info("Data transformed using preprocessor")

            train_arr = np.c_[X_train_transformed, y_train.to_numpy()]
            test_arr = np.c_[X_test_transformed, y_test.to_numpy()]
            logging.info("Train and Test arrays created")

            save_object(
                file_path= self.transformation_config.preprocessor_obj_file_path,
                obj= preprocessor_obj
            )

            return {
                'train_arr': train_arr,
                'test_arr': test_arr,
                'preprocessor_obj_file_path': self.transformation_config.preprocessor_obj_file_path
            }
            

        except Exception as e:
            raise CustomException(e, sys) from e
            logging.error(f"Error during data transformation: {e}")
        

           