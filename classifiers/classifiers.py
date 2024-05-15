# -*- coding: utf-8 -*-

# May require to install the following packages:
#!pip install tiktoken
#!pip install nltk
#!pip install spacy
#!pip install scipy

import nltk 
nltk.download('punkt')
nltk.download('wordnet')
import spacy
import tiktoken

from nltk.tokenize import word_tokenize
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
from spacy.lang.en.stop_words import STOP_WORDS as en_stop

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import  cross_val_score , GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier


import warnings
warnings.filterwarnings("ignore")

# Custom imports
from preprocessing import import_data_classifier


"""
    The goal of this script is to train 
    1 Logistic regression 
    2 Naive bayes 
    3 A mix of logistic and naive bayes 
    classifier models to predict the song genre (tag) based on lyrics data.

    First functions are utility functions to tokenize the data and split the data.
"""
def gpt_tokenize(doc):
    enc = tiktoken.encoding_for_model("gpt-4")
    tokens = enc.encode(doc)
    return [str(token) for token in tokens]


def split_data(train_set, validation_set):
    """
    Used to avoid redundancy in the code.
    """
    X_train = train_set['lyrics']
    Y_train = train_set['tag']
    X_validation = validation_set['lyrics']
    Y_validation = validation_set['tag']
    return X_train, Y_train, X_validation, Y_validation

def evaluate_model(model, X_train, Y_train, X_validation, Y_validation, scoring = 'accuracy'):
    """
    Evaluate the model with and without cross-validation.
    """
    # Evaluate the model
    accuracy = model.score(X_validation, Y_validation)
    print(scoring + " without cross validation:", accuracy)

    # Perform cross-validation in order to have a more robust evaluation of the model
   
    scores = cross_val_score(model, X_train, Y_train, cv=5, scoring=scoring, n_jobs = -1)
    print(scoring +  " after cross validation:")
    print(f"Mean {scoring}: {scores.mean()}")
    print(f"Standard deviation {scoring}: {scores.std()}")

    return accuracy, scores.mean()


"""         The following functions are the main functions to train the models.      """


def logistic_regression_classifier(file_path: str):
    print("Logistic regression classifier")
    """
    Train a logistic regression classifier model.
    """
    # Import the data
    train_set, validation_set = import_data_classifier(file_path)

    X_train, Y_train, X_validation, Y_validation = split_data(train_set, validation_set)
    # Train the model

    # basic model
    # model = make_pipeline(CountVectorizer(ngram_range = (1,1)), LogisticRegression())

    # After multiples empiric tries, the best parameters are:
    model = make_pipeline(CountVectorizer(tokenizer=gpt_tokenize, ngram_range=(1, 1),stop_words = list(en_stop)), StandardScaler(with_mean=False), LogisticRegression(max_iter=1000, solver='saga', penalty='l2'))
    print("model is fitting...")
    model.fit(X_train, Y_train)

    # Evaluate the model
    evaluate_model(model, X_train, Y_train, X_validation, Y_validation)

    return model



def naive_bayes_classifier(file_path :str):
    print("NB regression classifier")

    """
    Train a naive bayes classifier model.
    """
     # Import the data
    train_set, validation_set = import_data_classifier(file_path)

    X_train, Y_train, X_validation, Y_validation = split_data(train_set, validation_set)

    # Train the model

    # basic model
    # model = make_pipeline(CountVectorizer(ngram_range = (1,1)), MultinomialNB())

    # Using GridSearch to find the best hyperparameters
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(tokenizer=gpt_tokenize, stop_words=list(en_stop))),
        ('classifier', MultinomialNB())
    ])

    # Define the parameter grid for GridSearch
    param_grid = {
        'vectorizer__ngram_range': [(1, 1), (1, 2)],
        'classifier__alpha': [0.1, 0.5, 1.0]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, Y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print("Best parameters: ", best_params)

    # Evaluate the model
    evaluate_model(best_model, X_train, Y_train, X_validation, Y_validation)

    return best_model



def ensembling_classifier(file_path: str):
    # Import the data
    train_set, validation_set = import_data_classifier(file_path)
    X_train, Y_train, X_validation, Y_validation = split_data(train_set, validation_set)

    # Define the models
    LOG_model = logistic_regression_classifier(file_path)
    NB_model = naive_bayes_classifier(file_path)

    voting_clf = VotingClassifier(estimators=[('lr', LOG_model), ('nb', NB_model)], voting='soft')

    # Train the model directly on the vectorized data
    voting_clf.fit(X_train, Y_train)

    # Evaluate the model
    evaluate_model(voting_clf, X_train, Y_train, X_validation, Y_validation)

    return voting_clf


#logistic_regression_classifier("song_lyrics.csv")
#naive_bayes_classifier("song_lyrics.csv")
#ensembling_classifier("song_lyrics.csv")