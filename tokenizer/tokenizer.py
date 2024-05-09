import re
from collections import Counter


def get_word_count(lyrics):
    # Retirer les virgules et ce qui se trouve entre brackets et split par espaces
    lyrics = lyrics.lower()
    lyrics_no_brackets = re.sub("[\[].*?[\]]", "", lyrics)
    lyrics_no_punctuation = re.sub(r'[.,]', "", lyrics_no_brackets)

    words = lyrics_no_punctuation.split()
    counter = Counter(words)
    word_dict = {}

    for word, count in counter.items():
        word_dict[word] = {
            "count": count,
            "letters": [x for x in word]
        }

    return word_dict


def get_init_vocabulary(word_dict):
    vocab = set()

    for word in word_dict.keys():
        vocab.update(word_dict[word]["letters"])

    return vocab


def tokenize(word_dict, vocabulary):
    pairs = {}

    # Compute and store pairs and their frequency
    for value in word_dict.values():
        count = value["count"]
        letters = value["letters"]

        for i in range(len(letters) - 2):
            pair = "".join(letters[i:i+2])

            if pair not in pairs.values():
                pairs[pair] = count
            else:
                pairs[pair] += count

    # Get the most frequent pair
    best_pair = max(pairs, key=pairs.get)

    # Update vocabulary
    vocabulary.add(best_pair)

    # Update words decomposition
    for value in word_dict.values():
        letters = value["letters"]
        new_letters = []
        i = 0

        while i < len(letters):
            if "".join(letters[i:i+2]) == best_pair:
                new_letters.append(best_pair)
                i += 1
            else:
                new_letters.append(letters[i])
            i += 1

        value["letters"] = new_letters


def byte_pair_encoding(k, lyrics):
    # Récupérer la liste de mots, leur fréquence et leur décomposition en tokens
    word_count = get_word_count(test)

    # Récupérer le vocabulaire initial
    vocabulary = get_init_vocabulary(word_count)

    for i in range(k):
        tokenize(word_count, vocabulary)

    return vocabulary


# test = "[Verse 1: Cam'ron] Killa, Dipset Man I'll spit that pimp talk, you hang out where the pimp collide"
# vocab = byte_pair_encoding(10, test)
# print(vocab)
