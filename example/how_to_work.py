import pandas as pd
import numpy as np
import tensorflow as tf
from collections import Counter


vocab = Counter()

text = "Hi hi from Brazil bro"

for word in text.split(' '):
    word_lowercase = word.lower()
    vocab[word_lowercase] += 1

print(vocab)


def get_word_2_index(vocab):
    word2index = {}
    for i, word in enumerate(vocab):
        print(i, word)
        word2index[word] = i

    return word2index


word2index = get_word_2_index(vocab)

total_words = len(vocab)
matrix = np.zeros((total_words), dtype=float)

for word in text.split():
    matrix[word2index[word.lower()]] += 1
    print(matrix)

print("Hi from Brazil:", matrix)

matrix = np.zeros((total_words), dtype=float)
text = "Hi bro"
for word in text.split():
    matrix[word2index[word.lower()]] += 1

print("Hi:", matrix)