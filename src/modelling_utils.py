'''
This script aims to provide functions that will turn the modelling process easier
'''

'''
Importing libraries
'''

# Data manipulation and visualization.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Modelling, cleaning and preparation.
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# Debugging.
from src.exception import CustomException
import sys

# Warnings.
from warnings import filterwarnings
filterwarnings('ignore')

# Definições de cores -> todas estão numa escala de mais escura para mais clara.
CINZA1, CINZA2, CINZA3 = '#231F20', '#414040', '#555655'
CINZA4, CINZA5, CINZA6 = '#646369', '#76787B', '#828282'
CINZA7, CINZA8, CINZA9 = '#929497', '#A6A6A5', '#BFBEBE'
AZUL1, AZUL2, AZUL3, AZUL4 = '#174A7E', '#4A81BF', '#94B2D7', '#94AFC5'
VERMELHO1, VERMELHO2, VERMELHO3, VERMELHO4, VERMELHO5 = '#DB0527', '#E23652', '#ED8293', '#F4B4BE', '#FBE6E9'
VERDE1, VERDE2 = '#0C8040', '#9ABB59'
LARANJA1 = '#F79747'
AMARELO1, AMARELO2, AMARELO3, AMARELO4, AMARELO5 = '#FFC700', '#FFCC19', '#FFEB51', '#FFE37F', '#FFEEB2'
BRANCO = '#FFFFFF'

# Data cleaning.

def reg_ex(string, regex):
    '''
    Applies regular expression pattern matching to remove specified patterns from a string.

    Args:
        string (str): The input string to be processed.
        regex (str): The regular expression pattern to be matched and removed.

    Returns:
        str: The processed string with the specified patterns removed.

    Raises:
        CustomException: If an exception occurs during the regular expression compilation or substitution.
            The original exception is encapsulated within CustomException.
    '''
    try:
        compile = re.compile(regex)
        return compile.sub(r'', string)
    except Exception as e:
        raise CustomException(e, sys)
    

def lemmatize_with_pos(text, lemmatizer=WordNetLemmatizer()):
    '''
    Applies part-of-speech tagging and lemmatization to a given text.

    Args:
        text (str): The input text to be lemmatized.
        lemmatizer (object, optional): The lemmatizer object, default is WordNetLemmatizer.

    Returns:
        str: The lemmatized text.

    Raises:
        CustomException: If an exception occurs during tokenization, part-of-speech tagging, or lemmatization.
            The original exception is encapsulated within CustomException.
    '''
    try:
        # Tokenize the input text into individual words.
        word_tokens = word_tokenize(text)
        # Obtain a tuple with each token and its part-of-speech tag.
        pos_tags = pos_tag(word_tokens)
        # Apply lemmatization to each word considering the WordNetLemmatizer expects POS tags in specific format 'a' for adjective, 'v' for verb, 'n' for noun, etc.
        lemmatized_words = [lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag)) for word, tag in pos_tags]
        return ' '.join(lemmatized_words)
    except Exception as e:
        raise CustomException(e, sys)
    

def get_wordnet_pos(tag):
    '''
    Maps the POS tags from the Penn Treebank tagset to WordNet POS tags.

    Args:
        tag (str): The input POS tag from the Penn Treebank tagset.

    Returns:
        str: The corresponding WordNet POS tag.

    Raises:
        CustomException: If an exception occurs during the mapping.
            The original exception is encapsulated within CustomException.
    '''
    try:
        if tag.startswith('J'):
            return 'a'  # Adjective
        elif tag.startswith('V'):
            return 'v'  # Verb
        elif tag.startswith('N'):
            return 'n'  # Noun
        elif tag.startswith('R'):
            return 'r'  # Adverb
        else:
            return 'n'  # Default to Noun
    except Exception as e:
        raise CustomException(e, sys)
    

def nlp_data_cleaning(df):
    '''
    Performs data cleaning on a DataFrame with NLP text data.

    Args:
        df (pd.DataFrame): The input DataFrame containing a 'text' column to be cleaned.

    Returns:
        pd.DataFrame: The cleaned DataFrame with processed text.

    Raises:
        CustomException: If an exception occurs during the cleaning process.
            The original exception is encapsulated within CustomException.
    '''
    try:
        clean_df = df.copy()

        # Remove links and html tags.
        clean_df['clean_text'] = clean_df['text'].apply(lambda x: reg_ex(x, 'https?://\S+|www\.\S+'))
        clean_df['clean_text'] = clean_df['clean_text'].apply(lambda x: reg_ex(x, '<.*?>'))

        # Remove @user mentions as they don't add semantic meaning, introduce noise, and may violate privacy.
        clean_df['clean_text'] = clean_df['clean_text'].apply(lambda x: reg_ex(x, '@[\w]*'))

        # Remove punctuation, special characters, and numbers, converting to lowercase.
        clean_df['clean_text'] = clean_df['clean_text'].apply(lambda x: str.lower(reg_ex(x, '[^a-zA-Z# ]')))

        # Remove stop words.
        stop_words = stopwords.words('english')
        word_tokens = clean_df['clean_text'].apply(lambda x: word_tokenize(x)).tolist()
        text_without_stop_words = [' '.join([w for w in word_token_lst if w not in set(stop_words)]) for word_token_lst in word_tokens]
        clean_df['clean_text'] = text_without_stop_words

        # Apply lemmatization.
        lemmatizer = WordNetLemmatizer()
        clean_text = clean_df.clean_text.tolist()
        words_lemmatized_pos = [lemmatize_with_pos(text, lemmatizer) for text in clean_text]
        clean_df['clean_text'] = words_lemmatized_pos

        # Remove raw text column.
        clean_df = clean_df.drop(columns=['text'])

        return clean_df
    except Exception as e:
        raise CustomException(e, sys)