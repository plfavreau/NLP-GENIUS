from collections import Counter, defaultdict


def get_vocabulary(text):
    res = set()

    for word in text.split():
        res.update(word)

    return res


def get_corpus(text):
    # Initialize vocabulary with frequency of each word in text
    return Counter(text.split())


def get_stats(corpus):
    # Get frequency of adjacent symbol pairs (bigrams) in vocabulary
    pairs = defaultdict(int)

    for word, freq in corpus.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i+1]] += freq

    return pairs


def merge_corpus(pair, corpus):
    # Merge most frequent pair in all vocabulary words and update frequency
    merged_corpus = {}
    old = ' '.join(pair)
    replacement = ''.join(pair)

    for word in corpus:
        new_word = word.replace(old, replacement)
        merged_corpus[new_word] = corpus[word]

    return merged_corpus


# Sample text data
text = "Ce message est un test! On verra si ça marche."

vocabulary = get_vocabulary(text)
print("f", vocabulary)

# Convert each word in initial vocabulary to space-separated string of characters
corpus = get_corpus(text)
corpus = {' '.join(word): freq for word, freq in corpus.items()}
print("Initial corpus:", corpus)

# Number of BPE iterations
k = 10

for i in range(k):
    pairs = get_stats(corpus)

    if pairs:
        # Get the most frequent pair
        most_common_pair = max(pairs, key=pairs.get)
        corpus = merge_corpus(most_common_pair, corpus)
        vocabulary.update([''.join(most_common_pair)])

        print(f"After iteration {i+1}, Best pair: {most_common_pair}")
        print("Updated corpus:", corpus)
        print("f updated:", vocabulary)