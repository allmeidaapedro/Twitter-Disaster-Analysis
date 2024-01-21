'''
This script aims to create the predict pipeline for a simple web application which will be interacting with the pkl files, such that we can make predictions by giving values of input features. 
'''

# Debugging and verbose.
import sys
from src.logger import logging
from src.exception import CustomException
from src.artifacts_utils import load_object
from src.modelling_utils import nlp_data_cleaning

# Data manipulation.
import pandas as pd

# File handling.
import os


class PredictPipeline:
    '''
    Class for making predictions using a trained model and preprocessor.

    This class provides a pipeline for making predictions on new instances using a trained machine learning model and
    a preprocessor. It loads the model and preprocessor from files, preprocesses the input features, and estimates the probability of a tweet being related to a disaster.

    Methods:
        predict(features):
            Make predictions on new instances using the loaded model and preprocessor.

    Example:
        pipeline = PredictPipeline()
        new_features = [...]
        prediction = pipeline.predict(new_features)

    Note:
        This class assumes the availability of the load_object function.
    '''
    def __init__(self) -> None:
        '''
        Initializes a PredictPipeline instance.

        Initializes the instance. No specific setup is required in the constructor.
        '''
        pass


    def predict(self, tweet):
        '''
        Make predictions on new instances using the loaded model and preprocessor.

        Args:
            tweet: Input tweet for which we will estimate the probability of being related to a disaster (DataFrame).

        Returns:
            predictions: Tweet's probability of being related to a disaster.

        Raises:
            CustomException: If an exception occurs during the prediction process.
        '''
        try:


            model_path = os.path.join('artifacts', 'model.pkl')
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            
            logging.info('Load model and preprocessor objects.')

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            logging.info('Model and preprocessor succesfully loaded.')

            logging.info('Cleaning and preprocessing the input data (tweet).')

            clean_tweet = nlp_data_cleaning(tweet)
            prepared_data = preprocessor.transform(clean_tweet)

            logging.info('Input data (tweet) prepared for prediction.')

            logging.info('Predicting.')

            # Predict the tweet's probability of being related to a disaster.
            disaster_proba = model.predict_proba(prepared_data)
            disaster_proba = disaster_proba[:, 1][0]
            print(clean_tweet, disaster_proba)
            prediction = f"A probabilidade do tweet fornecido estar relacionado a um desastre é de {round(disaster_proba, 2)*100}%"
            print(prediction)
            logging.info("Tweet's probability of being related to a disaster successfully estimated. Prediction successfully made.")

            return prediction

        except Exception as e:
            raise CustomException(e, sys)
        

class InputData:
    '''
    Class for handling input data for predictions.

    This class provides a structured representation for input data (tweet) that is meant to be used for making predictions.
    It maps input tweet from HTML inputs to class attributes and provides a method to convert the input data (tweet) into
    a DataFrame format suitable for making predictions.

    Attributes:
        tweet (str): The tweet.

    Methods:
        get_input_data_df():
            Convert the mapped input data (tweet) into a DataFrame for predictions.

    Note:
        This class assumes the availability of the pandas library and defines the CustomException class.
    '''

    def __init__(self,
                 tweet: str) -> None:
        '''
        Initialize an InputData instance with mapped input data (tweet).

        Args:
            tweet (str): The tweet.
        '''
        
        # Map the tweet from html input.
        self.tweet = tweet

    
    def get_input_data_df(self):
        '''
        Convert the mapped input data (tweet) into a DataFrame for predictions.

        Returns:
            input_data_df (DataFrame): DataFrame containing the mapped input data.

        Raises:
            CustomException: If an exception occurs during the process.
        '''
        try:
            input_data_dict = dict()

            # Map the tweet to the form of a dataframe for being used in predictions.
            input_data_dict['text'] = [self.tweet]
            input_data_df = pd.DataFrame(input_data_dict)

            return input_data_df
        
        except Exception as e:
            raise CustomException(e, sys)