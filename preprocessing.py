import pandas as pd

def split_data(data: pd.DataFrame):
    """
    Split the data in a training and test set.
    The training set is composed of 80% of the data.
    The test set is composed of 20% of the data.
    Useful links: https://towardsdatascience.com/train-validation-and-test-sets-72cb40cba9e7
    """
    train_set = data.sample(frac=0.8, random_state=0)
    validation_set = data.drop(train_set.index)
    return train_set, validation_set

def import_data_classifier(file_path: str):
    """
    Preprocess the data for the classifier model.
    The transformation effectuated are:
    - Keep only the english songs and remove the language column
    - Remove the NaN values
    - Transform the text in lowercase
    - Split the data in a training and validation set
    """
    data = pd.read_csv(file_path, usecols=['lyrics', 'tag', 'language_cld3'])
    data = data.loc[data['language_cld3'] == 'en']
    data = data.drop(columns=['language_cld3'])
    data = data.dropna()
    data[['lyrics', 'tag']] = data[['lyrics', 'tag']].apply(lambda x: x.str.lower())
    train_set, validation_set = split_data(data)
    return train_set, validation_set

def import_data_n_gram(file_path: str):
    """
    Preprocess the data for the n-gram model.
    The transformation effectuated are:
    - Keep only the english songs and remove the language column
    - Remove the NaN values
    - Concatenate the title and the lyrics in a single column
    - Transform the text in lowercase
    - Split the data in a training and validation set
    Useful links: https://necromuralist.github.io/Neurotic-Networking/posts/nlp/n-gram-pre-processing/index.html
    """
    data = pd.read_csv(file_path, usecols=['title', 'lyrics', 'language_cld3'])
    data = data.loc[data['language_cld3'] == 'en']
    data = data.drop(columns=['language_cld3'])
    data = data.dropna()
    data['title_lyrics'] = data['title'] + ' ' + data['lyrics']
    data[['title_lyrics', 'title', 'lyrics']] = data[['title_lyrics', 'title', 'lyrics']].apply(lambda x: x.str.lower())
    train_set, validation_set = split_data(data)
    return train_set, validation_set

def import_data_transformer(file_path: str):
    """
    Preprocess the data for the transformer model.
    The transformation effectuated are:
    - Keep only the english songs and remove the language column
    - Remove the NaN values
    - Concatenate the title and the lyrics in a single column
    - Transform the text in lowercase
    - Split the data in a training and validation set
    Useful links: https://huggingface.co/docs/transformers/preprocessing
    """
    data = pd.read_csv(file_path, usecols=['title', 'lyrics', 'language_cld3'])
    data = data.loc[data['language_cld3'] == 'en']
    data = data.drop(columns=['language_cld3'])
    data = data.dropna()
    data['title_lyrics'] = data['title'] + ' ' + data['lyrics']
    data[['title_lyrics', 'title', 'lyrics']] = data[['title_lyrics', 'title', 'lyrics']].apply(lambda x: x.str.lower())
    train_set, validation_set = split_data(data)
    return train_set, validation_set