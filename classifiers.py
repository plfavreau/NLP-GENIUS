# -*- coding: utf-8 -*-

# May require to install the following packages:
#!pip install tiktoken
#!pip install nltk
#!pip install spacy
#!pip install scipy


import nltk 
""" nltk.download('punkt')
nltk.download('wordnet')
 """
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

# Tools imports
import optuna
import os

# Custom imports
from preprocessing import import_data_classifier


"""
    The goal of this script is to train classifier models to predict the song genre (tag) based on lyrics data.
    To do so, we will train three different models:
    1 Logistic regression 
    2 Naive bayes 
    3 A mix of logistic and naive bayes 
    
    Then, we will evaluate the models using cross-validation and accuracy score.

    Finally, we will use optuna to find best hyperparameters and check their impact on the perfomance.

    First functions are utility functions to tokenize the data and split the data.
"""
def gpt_tokenize(doc):
    enc = tiktoken.encoding_for_model("gpt-4")
    tokens = enc.encode(doc)
    # Normalize numerical tokens
    return ["<NUMBER>" if str(token).isnumeric() else str(token) for token in tokens]


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
# Import the data, and split it in a training and validation set
train_set, validation_set = import_data_classifier("song_lyrics.csv")
X_train, Y_train, X_validation, Y_validation = split_data(train_set, validation_set)

def logistic_regression_classifier(file_path: str):
    print("Logistic regression classifier")
    """
    Train a logistic regression classifier model.
    """
    # Train the model
    # This is a basic model :
    # model = make_pipeline(CountVectorizer(ngram_range = (1,1)), LogisticRegression())

    # After multiples empiric tries, the best parameters seems to be the following ones:
    #model = make_pipeline(CountVectorizer(tokenizer=gpt_tokenize, ngram_range=(1, 1),stop_words = list(en_stop)), StandardScaler(with_mean=False), LogisticRegression(max_iter=1000, solver='saga', penalty='l2'))

    # Using optuna results :
    model = make_pipeline(CountVectorizer(tokenizer=gpt_tokenize, ngram_range=(1, 1), stop_words=list(en_stop),token_pattern=None), StandardScaler(with_mean=False), LogisticRegression(max_iter=2000, solver='saga', penalty='elasticnet', l1_ratio=0.8119063905463674, C= 0.10275855119285837))
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
    # Define the models
    LOG_model = logistic_regression_classifier(file_path)
    NB_model = naive_bayes_classifier(file_path)

    voting_clf = VotingClassifier(estimators=[('lr', LOG_model), ('nb', NB_model)], voting='soft')

    # Train the model directly on the vectorized data
    voting_clf.fit(X_train, Y_train)

    # Evaluate the model
    evaluate_model(voting_clf, X_train, Y_train, X_validation, Y_validation)

    return voting_clf



def objective_logistic(trial):
    # Define the hyperparameters to be optimized
    l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
    #C = trial.suggest_loguniform('C', 1e-4, 1e4) deprecated
    C = trial.suggest_float('C', 1e-4, 1e4, log=True)

    model = make_pipeline(
        CountVectorizer(tokenizer=gpt_tokenize, ngram_range=(1, 1), stop_words=list(en_stop)),
        StandardScaler(with_mean=False),
        LogisticRegression(max_iter=1000, solver='saga', penalty='elasticnet', l1_ratio=l1_ratio, C=C)
    )

    model.fit(X_train, Y_train)
    accuracy = model.score(X_validation, Y_validation)
    return accuracy

def find_best_hp_with_optuna(classifier_type:int):
   # Database for the optuna dashboard
    if classifier_type == 1 : # Logistic regression
      storage_name = "optuna_logistic_regression.db"
    else: # Naive bayes
      storage_name = "optuna_naive_bayes.db"
    if os.path.exists(storage_name):
        os.remove(storage_name)

    # Create a study
    study = optuna.create_study(
        storage=f"sqlite:///{storage_name}",
        study_name=classifier_type,
        load_if_exists=False,
        direction="maximize",  # we want to maximize the score
    )
    if classifier_type == 1:
        study.optimize(objective_logistic, n_trials=50)
    else:
        study.optimize(objective_naive_bayes, n_trials=50)

    # print best trial
    print(f"Best value: {study.best_value} (params: {study.best_params})")
    for key, value in study.best_trial.params.items():
        if type(value) == float:
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")



def objective_naive_bayes(trial):
    alpha = trial.suggest_float('alpha', 0.01, 10.0)

    model = make_pipeline(
        CountVectorizer(tokenizer=gpt_tokenize, ngram_range=(1, 2), stop_words=list(en_stop)),
        MultinomialNB(alpha=alpha)
    )

    model.fit(X_train, Y_train)
    accuracy = model.score(X_validation, Y_validation)
    return accuracy


find_best_hp_with_optuna(classifier_type=2)
#logistic_regression_classifier("song_lyrics.csv")
#naive_bayes_classifier("song_lyrics.csv")
#ensembling_classifier("song_lyrics.csv")