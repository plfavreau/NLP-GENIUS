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

def import_data_transformer(file_path: str, nb_rows_to_load: int):
    """
    Preprocess the data for the transformer model.
    The transformation effectuated are:
    - Keep only the english songs and remove the language column
    - Remove the NaN values
    - Replace the square brackets by an empty string to remove the annotations and metadata
    - Keep only the songs with a title length inferior to 50 characters
    - Keep only the songs with a title composed of ASCII characters
    - Concatenate the title and the lyrics in a single column
    - Transform the text in lowercase
    - Split the data in a training and validation set
    Useful links: https://huggingface.co/docs/transformers/preprocessing
    """
    data = pd.read_csv(file_path, usecols=['title', 'lyrics', 'language_cld3'], nrows=nb_rows_to_load)
    data = data.loc[data['language_cld3'] == 'en']
    data = data.drop(columns=['language_cld3'])
    data = data.dropna()
    data['lyrics'] = data['lyrics'].str.replace(r"\[.*\]", "")
    data = data[data['title'].apply(lambda x: len(x) < 50)]
    data = data[data['title'].str.match(r'^[\x00-\x7F]*$')]
    data['title_lyrics'] = data['title'] + ' ' + data['lyrics']
    data[['title_lyrics', 'title', 'lyrics']] = data[['title_lyrics', 'title', 'lyrics']].apply(lambda x: x.str.lower())
    train_set, validation_set = split_data(data)
    return train_set, validation_set