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
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, auc, brier_score_loss
import time
import math

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
    

def evaluate_models_cv_classification(models, X_train, y_train, n_folds=5):
    '''
    Evaluate multiple machine learning models using stratified k-fold cross-validation (the stratified k-fold is useful for dealing with target imbalancement).

    This function evaluates a dictionary of machine learning models by training each model on the provided training data
    and evaluating their performance using stratified k-fold cross-validation. The evaluation metric used is ROC-AUC score.

    Args:
        models (dict): A dictionary where the keys are model names and the values are instantiated machine learning model objects.
        X_train (array-like): The training feature data.
        y_train (array-like): The corresponding target labels for the training data.

    Returns:
        pd.DataFrame: A DataFrame containing the evaluation results for each model, including their average validation scores
                  and training scores.

    Raises:
        CustomException: If an error occurs while evaluating the models.

    '''


    try:
        # Stratified KFold in order to maintain the target proportion on each validation fold - dealing with imbalanced target.
        stratified_kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        # Dictionaries with validation and training scores of each model for plotting further.
        models_val_scores = dict()
        models_train_scores = dict()

        for model in models:
            # Getting the model object from the key with his name.
            model_instance = models[model]

            # Measuring training time.
            start_time = time.time()
            
            # Fitting the model to the training data.
            model_instance.fit(X_train, y_train)

            end_time = time.time()
            training_time = end_time - start_time

            # Make predictions on training data and evaluate them.
            y_train_pred = model_instance.predict(X_train)
            train_score = roc_auc_score(y_train, y_train_pred)

            # Evaluate the model using k-fold cross validation, obtaining a robust measurement of its performance on unseen data.
            val_scores = cross_val_score(model_instance, X_train, y_train, scoring='roc_auc', cv=stratified_kfold)
            avg_val_score = val_scores.mean()
            val_score_std = val_scores.std()

            # Adding the model scores to the validation and training scores dictionaries.
            models_val_scores[model] = avg_val_score
            models_train_scores[model] = train_score

            # Printing the results.
            print(f'{model} results: ')
            print('-'*50)
            print(f'Training score: {train_score}')
            print(f'Average validation score: {avg_val_score}')
            print(f'Standard deviation: {val_score_std}')
            print(f'Training time: {round(training_time, 5)} seconds')
            print()

        # Plotting the results.
        print('Plotting the results: ')

        # Converting scores to a dataframe
        val_df = pd.DataFrame(list(models_val_scores.items()), columns=['Model', 'Average Val Score'])
        train_df = pd.DataFrame(list(models_train_scores.items()), columns=['Model', 'Train Score'])
        eval_df = val_df.merge(train_df, on='Model')

        # Sorting the dataframe by the best ROC-AUC score.
        eval_df  = eval_df.sort_values(['Average Val Score'], ascending=False).reset_index(drop=True)

        # Plotting each model and their train and validation (average) scores.
        fig, ax = plt.subplots(figsize=(20, 6))
        width = 0.35

        x = np.arange(len(eval_df['Model']))
        y = np.arange(len(eval_df['Train Score']))

        val_bars = ax.bar(x - width/2, eval_df['Average Val Score'], width, label='Average Validation Score', color=LARANJA1)
        train_bars = ax.bar(x + width/2, eval_df['Train Score'], width, label='Train Score', color=AZUL1)

        ax.set_xlabel('Model', color=CINZA1, labelpad=20, fontsize=10.8)
        ax.set_ylabel('ROC-AUC Score', color=CINZA1, labelpad=20, fontsize=10.8)
        ax.set_title("Models' Performances", fontweight='bold', fontsize=12, pad=30, color=CINZA1)
        ax.set_xticks(x, eval_df['Model'], rotation=0, color=CINZA1, fontsize=10.8)
        ax.tick_params(axis='y', color=CINZA1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(CINZA9)
        ax.spines['bottom'].set_color(CINZA9)

        # Add scores on top of each bar
        for bar in val_bars + train_bars:
            height = bar.get_height()
            plt.annotate('{}'.format(round(height, 2)),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha='center', va='bottom', color=CINZA1)

        ax.legend(loc='upper left')

        return eval_df
    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_classifier(y_true, y_pred, probas):
    '''
    Evaluate the performance of a binary classifier and visualize results.

    This function calculates and displays various evaluation metrics for a binary classifier,
    including the classification report, confusion matrix, ROC AUC curve, PR curve and brier score.

    Args:
    - y_true: True binary labels.
    - y_pred: Predicted binary labels.
    - probas: Predicted probabilities of positive class.

    Returns:
    - None (displays evaluation metrics).

    Raises:
    - CustomException: If an error occurs during evaluation.
    '''

    try:
        # Classification report.
        print(classification_report(y_true, y_pred))

        # Calculating and printing brier score.
        brier_score = brier_score_loss(y_true, probas)
        print(f'Brier Score: {round(brier_score, 2)}')
        
        # Confusion matrix.
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot = True, fmt = 'd')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Values')
        plt.ylabel('Real Values')
        plt.show()
        
        # ROC AUC Curve and score.
        fpr, tpr, thresholds = roc_curve(y_true, probas)
        roc_auc = roc_auc_score(y_true, probas)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.2f}', color=AZUL1)
        ax.plot([0, 1], [0, 1], linestyle='--', color=CINZA4)  # Random guessing line.
        ax.set_xlabel('False Positive Rate', fontsize=10.8, color=CINZA1, labelpad=20)
        ax.set_ylabel('True Positive Rate', fontsize=10.8, color=CINZA1, labelpad=20)
        ax.set_xticks(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], color=CINZA1)
        ax.set_yticks(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], color=CINZA1)
        ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontweight='bold', fontsize=12, color=CINZA1, pad=20)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(CINZA9)
        ax.spines['bottom'].set_color(CINZA9)
        ax.legend()
    
        # PR AUC Curve and score.

        # Calculate model precision-recall curve.
        precision, recall, _ = precision_recall_curve(y_true, probas)
        pr_auc = auc(recall, precision)

        fig, ax = plt.subplots(figsize=(7, 4))
        # Plot the model precision-recall curve.
        ax.plot(recall, precision, marker='.', label=f'PR AUC = {pr_auc:.2f}', color=AZUL1)
        ax.set_xlabel('Recall', fontsize=10.8, color=CINZA1, labelpad=20)
        ax.set_ylabel('Precision', fontsize=10.8, color=CINZA1, labelpad=20)
        ax.set_xticks(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], color=CINZA1)
        ax.set_yticks(ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], color=CINZA1)
        ax.set_title('Precision-Recall (PR) Curve', fontweight='bold', fontsize=12, color=CINZA1, pad=20)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(CINZA9)
        ax.spines['bottom'].set_color(CINZA9)
        ax.legend()


    except Exception as e:
        raise CustomException(e, sys)
    

def plot_probability_distributions(y_true, probas):
    '''
    Plots the kernel density estimate (KDE) of predicted probabilities for disaster related and non-related tweets.

    Parameters:
    - y_true (array-like): The true class labels (0 for not related tweets, 1 for disaster related tweets).
    - probas (array-like): Predicted probabilities for the positive class (disaster related tweets).

    Raises:
    - CustomException: Raised if an unexpected error occurs during plotting.

    Example:
    ```python
    plot_probability_distributions(y_true, probas)
    ```

    Dependencies:
    - pandas
    - seaborn
    - matplotlib

    Note:
    - The function assumes the existence of color constants VERDE1, VERMELHO1, CINZA1, CINZA9.

    The function creates a KDE plot illustrating the distribution of predicted probabilities for disaster related and non-related tweets.
    It provides visual insights into the model's ability to distinguish between the two classes.

    '''
    try:
        probas_df = pd.DataFrame({'Probabilidade de Tweet Estar Relacionado a Desastre': probas,
                                'Tweet Relacionado a Desastre': y_true})

        fig, ax = plt.subplots(figsize=(10, 4))
        sns.kdeplot(data=probas_df, x='Probabilidade de Tweet Estar Relacionado a Desastre', hue='Tweet Relacionado a Desastre', fill=True, ax=ax, palette=[VERDE1, VERMELHO1])
        ax.set_title('Distribuição das Probabilidades Preditas entre Tweets Relacionados a Desastres e Não Relacionados', fontweight='bold', fontsize=12, color=CINZA1, pad=20)
        ax.set_xlabel('Probabilidades Preditas', fontsize=10.8, color=CINZA1, labelpad=20)
        ax.set_ylabel('Densidade', fontsize=10.8, color=CINZA1, labelpad=20)
        ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'],
                    color=CINZA1)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6],
                    ['0.0', '0.2', '0.4', '0.6', '0.8', '1.0', '1.2', '1.4', '1.6'],
                    color=CINZA1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(CINZA9)
        ax.spines['bottom'].set_color(CINZA9)
    except Exception as e:
        raise CustomException(e, sys)


def plot_scores_percentages(y_true, probas):
    '''
    Plots the percentage of disaster related and non-relared tweets grouped by predicted probability score ranges.

    Parameters:
    - y_true (array-like): The true class labels (0 for non-related tweets, 1 for disaster related tweets).
    - probas (array-like): Predicted probabilities for the positive class (disaster related tweets).

    Raises:
    - CustomException: Raised if an unexpected error occurs during plotting.

    Example:
    ```python
    plot_scores_percentages(y_true, probas)
    ```

    Dependencies:
    - pandas
    - seaborn
    - matplotlib

    The function creates a horizontal stacked bar chart illustrating the percentage of disaster related and non-related tweets
    for different predicted probability score ranges. It helps visualize the distribution of predicted probabilities.

    Note:
    - The function assumes the existence of color constants VERDE1, VERMELHO1, CINZA1.

    '''
    try:
        probas_df = pd.DataFrame({'Probabilidade de Tweet Estar Relacionado a Desastre': probas,
                                'Tweet Relacionado a Desastre': y_true}).reset_index(drop=True)
        thresholds = np.arange(0.0, 1.1, 0.1)
        labels = [f'{t:.1f} a {t + 0.1:.1f}' for t in thresholds[:-1]]
        probas_df['Faixa de Score'] = pd.cut(probas_df['Probabilidade de Tweet Estar Relacionado a Desastre'], bins=thresholds, labels=labels, include_lowest=True)

        probas_grouped = probas_df.groupby(['Faixa de Score'])[['Tweet Relacionado a Desastre']].mean().reset_index().rename(columns={'Tweet Relacionado a Desastre': 'Disaster Related Tweet'})
        probas_grouped['Disaster Non-Related Tweet'] = 1 - probas_grouped['Disaster Related Tweet']

        fig, ax = plt.subplots(figsize=(15, 4))

        # Plot the horizontal stacked bar chart
        sns.barplot(x="Disaster Non-Related Tweet", y="Faixa de Score", data=probas_grouped, color=VERDE1, label="Tweet NÃO Relacionado a Desastre", ax=ax, left=probas_grouped['Disaster Related Tweet'])
        sns.barplot(x="Disaster Related Tweet", y="Faixa de Score", data=probas_grouped, color=VERMELHO1, label="Tweet Relacionado a Desastre", ax=ax)

        ax.set_title('Percentual de Tweets Relacionados a Desastres e Não Relacionados por Faixa de Score', color=CINZA1, fontweight='bold', fontsize=12, pad=20)
        ax.set_yticks(ticks=range(10), labels=labels, color=CINZA1, fontsize=11.2)
        ax.set_ylabel('')
        ax.get_xaxis().set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # Annotate the percentages inside the bars
        for i, bar in enumerate(ax.patches):
            width = bar.get_width()
            height = bar.get_height()
            x = bar.get_x()
            y = bar.get_y()
            
            if i < len(probas_grouped):
                continue
            
            percentage = width * 100  
            ax.text(x + width / 2, y + height / 2, f"{percentage:.1f}%", ha="center", va="center", color="white", fontsize=10.8)

        for i, bar in enumerate(ax.patches[:len(probas_grouped)]):
            width = bar.get_width()
            height = bar.get_height()
            x = bar.get_x()
            y = bar.get_y()
            
            percentage = probas_grouped['Disaster Non-Related Tweet'].iloc[i] * 100  
            ax.text(x + width / 2, y + height / 2, f"{percentage:.1f}%", ha="center", va="center", color="white", fontsize=10.8)

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", ncol=1)
        plt.show()
    except Exception as e:
        raise CustomException(e, sys)