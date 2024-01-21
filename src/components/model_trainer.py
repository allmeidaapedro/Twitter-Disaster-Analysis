'''
This script aims to train and save the selected final model from the modelling notebook.
'''

'''
Importing the libraries
'''

# File handling.
import os
from dataclasses import dataclass

# Debugging and verbose.
import sys
from src.exception import CustomException
from src.logger import logging

# Modelling.
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

# Utils.
from src.artifacts_utils import save_object


@dataclass
class ModelTrainerConfig:
    '''
    Configuration class for model training.

    This data class holds configuration parameters related to model training. It includes attributes such as
    `model_file_path` that specifies the default path to save the trained model file.

    Attributes:
        model_file_path (str): The default file path for saving the trained model. By default, it is set to the
                              'artifacts' directory with the filename 'model.pkl'.

    Example:
        config = ModelTrainerConfig()
        print(config.model_file_path)  # Output: 'artifacts/model.pkl'

    Note:
        This class uses the @dataclass decorator to automatically generate special methods like __init__ and __repr__
        based on the defined attributes.
    '''

    model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    '''
    This class is responsible for training and saving the best Logistic Regression model from modelling notebook.

    Attributes:
        model_trainer_config (ModelTrainerConfig): An instance of `ModelTrainerConfig` for configuration settings.

    Methods:
        apply_model_trainer(X_train_prepared, X_test_prepared, y_train, y_test):
            Trains the best Logistic Regression model using the provided prepared training and testing data,
            and returns ROC AUC and classification report on the test set.
    '''

    def __init__(self) -> None:
        '''
        Initializes a new instance of the `ModelTrainer` class.

        Attributes:
            model_trainer_config (ModelTrainerConfig): An instance of `ModelTrainerConfig` for configuration settings.
        '''
        self.model_trainer_config = ModelTrainerConfig()
    
    
    def apply_model_trainer(self, X_train_prepared, X_test_prepared, y_train, y_test):
        '''
        Trains the best Logistic Regression model using the provided prepared training and testing data, 
        the best hyperparameters found during the modelling notebook using k-fold cross validation and bayesian optimization and returns the ROC AUC score, and classification report on the test set.

        Args:
            X_train_prepared (np sparse matrix): The prepared predictor training data.
            X_test_prepared (np sparse matrix): The prepared predictor testing data.
            y_train (pd.series): The target training data.
            y_test (pd.series): The target testing data.

        Returns:
            float: The ROC AUC score and classification report of the best model on the test set.

        Raises:
            CustomException: If an error occurs during the training and evaluation process.
        '''

        try:
            logging.info('Started to train the best Logistic Regression model.')

            best_params = { 'penalty': 'l2',
                            'tol': 0.0123285564509487,
                            'C': 1.2348858060072314,
                            'max_iter': 1000, 
                            'warm_start': False}
            
            best_model = LogisticRegression(**best_params)

            best_model.fit(X_train_prepared, y_train)

            logging.info('Saving the best model.')

            save_object(
                file_path=self.model_trainer_config.model_file_path,
                object=best_model
            )

            logging.info('Best model ROC AUC, and classification report on test set returned.')

            y_pred = best_model.predict(X_test_prepared)
            probas = best_model.predict_proba(X_test_prepared)[:, 1]

            roc_auc = roc_auc_score(y_test, probas)
            class_report = classification_report(y_test, y_pred)

            return roc_auc, class_report

        except Exception as e:
            raise CustomException(e, sys)