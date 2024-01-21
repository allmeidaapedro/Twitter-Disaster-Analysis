'''
This script aims to apply data preparation.
'''

# Debugging and verbose.
import sys
from src.exception import CustomException
from src.logger import logging

# File handling.
import os

# Data manipulation.
import numpy as np
import pandas as pd
from dataclasses import dataclass

# Data preparation.
from sklearn.feature_extraction.text import CountVectorizer

# Utils.
from src.artifacts_utils import save_object


@dataclass
class DataTransformationConfig:
    '''
    Configuration class for data transformation.

    This data class holds configuration parameters related to data transformation. It includes attributes such as
    `preprocessor_file_path` that specifies the default path to save the preprocessor object file.

    Attributes:
        preprocessor_file_path (str): The default file path for saving the preprocessor object. By default, it is set
                                     to the 'artifacts' directory with the filename 'preprocessor.pkl'.

    Example:
        config = DataTransformationConfig()
        print(config.preprocessor_file_path)  # Output: 'artifacts/preprocessor.pkl'

    Note:
        This class uses the @dataclass decorator to automatically generate special methods like __init__ and __repr__
        based on the defined attributes.
    '''

    preprocessor_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    '''
    Data transformation class for preprocessing and transformation of train and test sets.

    This class handles the text representation in numerical vectors for nlp task, including Bag of Words approach with CountVectorizer.

    :ivar data_transformation_config: Configuration instance for data transformation.
    :type data_transformation_config: DataTransformationConfig
    '''
    def __init__(self) -> None:
        '''
        Initialize the DataTransformation instance with a DataTransformationConfig.
        '''
        self.data_transformation_config = DataTransformationConfig()


    def get_preprocessor(self):
        '''
        Get a preprocessor for data transformation.

        This method sets up the Bag of Words approach for text representation in numerical vectors with CountVectorizer.

        :return: Preprocessor object for data transformation.
        :rtype: ColumnTransformer
        :raises CustomException: If an exception occurs during the preprocessing setup.
        '''

        try:
            # Bag of Words.
            preprocessor = CountVectorizer( analyzer='word',
                                            stop_words='english',
                                            ngram_range=(1, 3),
                                            token_pattern=r'\w{1,}',
                                            )
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
    
    
    def apply_data_transformation(self, train_path, test_path):
        '''
        Apply data transformation process.

        Reads, preprocesses, and transforms training and testing datasets.

        :param train_path: Path to the training dataset CSV file.
        :param test_path: Path to the test dataset CSV file.
        :return: Prepared training and testing predictor and target datasets and the preprocessor file path.
        :rtype: tuple
        :raises CustomException: If an exception occurs during the data transformation process.
        '''
        
        try:

            logging.info('Reading train and test sets.')

            # Obtaining train and test entire sets from artifacts.
            train = pd.read_csv(train_path)
            test = pd.read_csv(test_path)

            logging.info('Obtaining preprocessor object.')

            preprocessor = self.get_preprocessor()

            # Getting train and test predictor and target sets.
            X_train = train['clean_text'].copy()
            y_train = train['target'].copy()

            X_test = test['clean_text'].copy()
            y_test = test['target'].copy()

            logging.info('Preprocessing train and test sets - Bag of Words, CountVectorizer.')

            X_train_prepared = preprocessor.fit_transform(X_train, y_train)
            X_test_prepared = preprocessor.transform(X_test)


            logging.info('Train and test sets prepared.')

            logging.info('Save preprocessing object.')

            save_object(
                file_path=self.data_transformation_config.preprocessor_file_path,
                object=preprocessor
            )
        
            return X_train_prepared, X_test_prepared, y_train, y_test, self.data_transformation_config.preprocessor_file_path
        
        except Exception as e:
            raise CustomException(e, sys)
        