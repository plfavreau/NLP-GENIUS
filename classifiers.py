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
from sklearn.metrics import accuracy_score , classification_report, confusion_matrix
from sklearn.ensemble import VotingClassifier


# Tools imports
import optuna
import os
import matplotlib.pyplot as plt
import seaborn as sns
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
""" def gpt_tokenize(doc):
    enc = tiktoken.encoding_for_model("gpt-4")
    tokens = enc.encode(doc)
    # Normalize numerical tokens
    return ["<NUMBER>" if str(token).isnumeric() else str(token) for token in tokens]
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
   
    """ scores = cross_val_score(model, X_train, Y_train, cv=5, scoring=scoring, n_jobs = -1)
    print(scoring + " after cross validation:")
    print("Mean %s: %.2f, Standard deviation %s: %.2f" % (scoring, scores.mean(), scoring, scores.std()))
    print("Worst score: %.2f" % scores.min())
    print("Best score: %.2f" % scores.max()) 


    return accuracy, scores.mean()"""



"""         The following functions are the main functions to train the models.      """
# Import the data, and split it in a training and validation set
#train_set, validation_set = import_data_classifier("song_lyrics.csv")
#X_train, Y_train, X_validation, Y_validation = split_data(train_set, validation_set)
from sklearn.model_selection import train_test_split
import pandas as pd
df = pd.read_csv("song_lyrics.csv", skiprows=lambda i: i % 100 != 0)
df = df[df['tag'] != 'misc']
if 'language' in df.columns:
    df = df[df['language'] == 'en']
df = df[['title', 'lyrics', 'tag']]
df.reset_index(drop=True, inplace=True)
df = df.sample(frac = 1)
X = df['lyrics']
Y = df['tag']

# Split the data into training and test sets (80% training, 20% test)
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
def logistic_regression_classifier(file_path: str, eval=True):
    print("Logistic regression classifier")
    """
    Train a logistic regression classifier model.
    """
    # Train the model
    # This is a basic model :
    model = make_pipeline(CountVectorizer(ngram_range = (1,1)), LogisticRegression(max_iter=1000, solver='saga', l1_ratio=0.811, penalty='elasticnet', C=0.127)) 

    # After multiples empiric tries, the best parameters seems to be the following ones:
    #model = make_pipeline(CountVectorizer(tokenizer=gpt_tokenize, ngram_range=(1, 1),stop_words = list(en_stop)), StandardScaler(with_mean=False), LogisticRegression(max_iter=1000, solver='saga', penalty='l2'))

    # Using optuna results :
    # model = make_pipeline(CountVectorizer(tokenizer=gpt_tokenize, ngram_range=(1, 1), stop_words=list(en_stop),token_pattern=None), StandardScaler(with_mean=False), LogisticRegression(max_iter=1000, solver='saga', penalty='elasticnet', l1_ratio=0.8119063905463674, C= 0.10275855119285837))
    print("model is fitting...")
    model.fit(X_train, Y_train)
    #Y_pred = model.predict(X_validation)
    print(model.score(X_test, Y_test))
    #print(classification_report(Y_validation, Y_pred))
    # Evaluate the model
    if eval:
        evaluate_model(model, X_train, Y_train, X_validation, Y_validation)

    return model



def plot_confusion_matrix(conf_matrix, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title(title)
    plt.show()

def naive_bayes_classifier(file_path :str, eval=True):
    print("NB regression classifier")

    # Train the model
    model = make_pipeline(CountVectorizer(ngram_range=(1,2)), MultinomialNB(alpha=0.8295352927506775)) # Default alpha = 1
    model.fit(X_train, Y_train)
    #print(model.score(X_test, Y_test))
    

    # Using GridSearch to find the best hyperparameters
    """ pipeline = Pipeline([
        ('vectorizer', CountVectorizer(tokenizer=gpt_tokenize, stop_words=list(en_stop))),
        ('classifier', MultinomialNB(force_alpha=True))
    ])

    param_grid = {
        'vectorizer__ngram_range': [(1, 1), (1, 2)],
        'classifier__alpha': [0.1, 0.5, 1.0] # todo test with multiples values
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, Y_train)

    best_params = grid_search.best_params_
    print("Best parameters: ", best_params) """

    # Accuracy on validation set
    """ if eval:
        # Evaluate accuracy
        accuracy = model.score(X_validation, Y_validation)
        print("Accuracy without Grid Search:", accuracy)
        
        # Other metrics
        Y_pred = model.predict(X_validation)
        print("Classification Report:")
        print(classification_report(Y_validation, Y_pred))

        print("Confusion Matrix:")
        conf_matrix_before = confusion_matrix(Y_validation, Y_pred)
        print(conf_matrix_before)
        plot_confusion_matrix(conf_matrix_before, "Confusion Matrix before Grid Search")
        
        accuracy = model.score(X_validation, Y_validation)
        print("Accuracy with best parameters:", accuracy)

        # Other metrics
        Y_pred = model.predict(X_validation)
        print("Classification Report:")
        print(classification_report(Y_validation, Y_pred))

        print("Confusion Matrix:")
        conf_matrix_after = confusion_matrix(Y_validation, Y_pred)
        print(conf_matrix_after)
        plot_confusion_matrix(conf_matrix_after, "Confusion Matrix after Grid Search") """

    return model


def ensembling_classifier(file_path: str):
    # Define the models
    LOG_model = logistic_regression_classifier(file_path, eval=False)
    NB_model = naive_bayes_classifier(file_path, eval=False)

    voting_clf = VotingClassifier(estimators=[('lr', LOG_model), ('nb', NB_model)], voting='hard')

    # Train the model directly on the vectorized data
    voting_clf.fit(X_train, Y_train)

    # Evaluate the model
    print(voting_clf.score(X_validation, Y_validation))
    print("Ensemble classifier accuracy:")
    print(classification_report(Y_validation, voting_clf.predict(X_validation)))
    #evaluate_model(voting_clf, X_train, Y_train, X_validation, Y_validation)

    return voting_clf



def objective_logistic(trial):
    # Define the hyperparameters to be optimized
    l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
    C = trial.suggest_float('C', 1e-4, 1e4, log=True)

    model = make_pipeline(
        CountVectorizer(tokenizer=gpt_tokenize, ngram_range=(1, 1), stop_words=list(en_stop)),
        StandardScaler(with_mean=False),
        LogisticRegression(max_iter=2000, solver='saga', penalty='elasticnet', l1_ratio=l1_ratio, C=C)
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

# define main
if __name__ == "__main__":
    df_test = pd.read_csv("song_lyrics.csv", skiprows=lambda i: i % 978 != 0 , nrows=10) # Change here to test different values

    df_test = df_test[df_test['tag'] != 'misc']
    if 'language' in df_test.columns:
        df_test = df_test[df_test['language'] == 'en']
    df_test = df_test[['title', 'lyrics', 'tag']]
    df_test.reset_index(drop=True, inplace=True)


    model = logistic_regression_classifier("song_lyrics.csv", eval=False)
    for song_name, song_lyrics, song_tag in zip(df_test['title'], df_test['lyrics'], df_test['tag']):
        print("Song:", song_name)
        print("Tag:", song_tag)
        # Convert the lyrics to a list and predict probabilities
        probabilities = model.predict_proba([song_lyrics])

        # Print the distribution of probabilities
        print("Distribution of Probabilities:")
        for class_label, probability in zip(model.classes_, probabilities[0]):
            if(probability > 0.0001):
                print(f"{class_label}: {probability:.4f}")
        max_prob_index = probabilities.argmax()
        predicted_class = model.classes_[max_prob_index]
        if predicted_class != song_tag:
            print(f'Model failed to predict. Actual tag is {song_tag}, predicted tag is {predicted_class}')
        print()
#find_best_hp_with_optuna(classifier_type=2)


#logistic_regression_classifier("song_lyrics.csv", eval=False)
#naive_bayes_classifier("song_lyrics.csv", eval=False)
#ensembling_classifier("song_lyrics.csv")
# Using grid search 
# Best parameters:  {'classifier__alpha': 0.1, 'vectorizer__ngram_range': (1, 1)}
# Accuracy with best parameters: 0.5632780082987552
# Using optuna
# Best value: 0.40174866627148786 (params: {'alpha': 7.862897656446976})
# alpha: 7.86
"""Best value: 0.5484588026081803 (params: {'l1_ratio': 0.8717932830416162, 'C': 70.42865780156407})
l1_ratio: 0.87
C: 70.43"""


""" import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import LabelEncoder
vectorizer = CountVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_validation)


Y_train_vec = vectorizer.fit_transform(Y_train)
Y_test_vec = vectorizer.transform(Y_validation)
# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_vec.toarray(), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_vec.toarray(), dtype=torch.float32)

label_encoder = LabelEncoder()
Y_train_indices = label_encoder.fit_transform(Y_train)
Y_test_indices = label_encoder.transform(Y_validation)

Y_train_tensor = torch.tensor(Y_train_indices, dtype=torch.long)
Y_test_tensor = torch.tensor(Y_test_indices, dtype=torch.long)
# Define logistic regression model
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
        self.softmax = nn.Softmax(dim=1)
        #self.relu = nn.ReLU()

    def forward(self, x):
        out = self.linear(x)
        #out = self.relu(out)
        out = self.softmax(out)
        #out = torch.sigmoid(out)
        return out

input_size = X_train_tensor.shape[1]
num_classes = len(Y_train.unique())
# Initialize the model
model = LogisticRegression(input_size, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
print(X_train_tensor.shape)
print(X_test_tensor.shape)
print(Y_train_tensor.shape)
num_epochs = 200
for epoch in range(num_epochs):
    # Training phase
    model.train()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, Y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Validation phase
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test_tensor)
        _, predicted_val = torch.max(val_outputs, dim=1)
        num_correct = (predicted_val == Y_test_tensor).sum().item()
        accuracy = num_correct / Y_test_tensor.size(0) * 100

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%')

    model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)

cm = confusion_matrix(Y_test_tensor.numpy(), predicted.numpy())
print(cm)
report = classification_report(Y_test_tensor.numpy(), predicted.numpy())
print(report) """