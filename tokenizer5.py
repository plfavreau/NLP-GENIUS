
def init_corpus(data):
    """
    Set the initial corpus as all the words present in data wth their frequency
    """
    corpus = {}
    data_split = [x.split() for x in data]

    for datum in data_split:
        for word in datum:
            if word in corpus:
                corpus[word] += 1
            else:
                corpus[word] = 1

    return corpus


def init_vocabulary(corpus):
    """
    corpus: list of words
    Set the initial vocabulary as all the unique characters present in data
    """

    #shadow_char = [' ', '[', ']', '(', ')', '"', ',', '.', ';']
    shadow_char = []
    vocabulary = set()

    for word in corpus.keys():
        for char in word:
            if char not in shadow_char:
                vocabulary.add(char)

    return vocabulary


def init_pairs(corpus):
    """
    Init a dictionary containing all the pairs found in the corpus with their frequency
    (a, b): 3
    (m, e): 2
    (he, y) 6
    """
    pairs = {}

    for word, freq in corpus.items():
        for i in range(len(word) - 1):
            p = word[i:i + 2]

            if p in pairs:
                pairs[p] += freq
            else:
                pairs[p] = freq

    return pairs


def byte_pair_encoding(data, k):
    """
    data: list of all lyrics
    k: number of merges
    Apply the BPE algorithm on the specified set for k merges.
    """
    corpus = init_corpus(data)
    vocabulary = init_vocabulary(corpus)
    pairs = init_pairs(corpus)

    for i in range(k):
        most_common_pair = max(vocabulary)
        vocabulary.update([''.join(most_common_pair)])

        pair = find_pair()
        merged_pair = merge_pair()
        vocabulary.update([merged_pair])

    return vocabulary
