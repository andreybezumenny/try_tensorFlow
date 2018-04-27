from keras.models import Sequential
from keras import optimizers
from keras.layers import Dense, Embedding, LSTM
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from collections import Counter
import helpers
import numpy as np
import pandas as pn


datasets = pn.read_csv('datasets/train.csv')

n_classes = 2
data = datasets['0'].tolist()
intent = datasets['1'].tolist()

test_data = helpers.get_synonims(data)
test_intent = intent

X_train, X_test, y_train, y_test = train_test_split(data, intent, test_size=0.5)


def get_vocab(train, test):
    vocab = Counter()

    for text in train:
        for word in text.split(' '):
            vocab[word.lower()] += 1

    for text in test:
        for word in text.split(' '):
            vocab[word.lower()] += 1
    return vocab


vocab = get_vocab(X_train, X_test)
total_words = len(vocab)

print("Total words:", total_words)


def get_word_2_index(vocab):
    word2index = {}
    for i, word in enumerate(vocab):
        word2index[word.lower()] = i
    return word2index


vocab = get_vocab(X_train, X_test)
word2index = get_word_2_index(vocab)

print('Index of the word \'the\'', word2index['the'])


def get_batch(data, intent):
    batches = []
    results = []
    texts = data
    categories = intent
    iter1 = 0 # this iterator only for count keyError
    iter2 = 0
    for text in texts:
        layer = np.zeros(total_words, dtype=float)
        for word in text.split():
            try:
                layer[word2index[word.lower()]] += 1
            except KeyError:
                iter1 = iter1 + 1
                print('KeyError with word:', word, iter1)
            iter2 = iter2 + 1
            print('num word ',iter2)
        batches.append(layer)

    for category in categories:
        y = np.zeros(n_classes, dtype=float)
        y[category] = 1.0

        results.append(y)

    return np.array(batches), np.array(results)


X_train_np, y_train_np = get_batch(data, intent)
X_test_np, y_test_np = get_batch(test_data, test_intent)

model_path = "save_model/4566_binary_model.json"
weights_path = "save_model/4566_binary_model.h5"
max_features = total_words  # 5000
batch_size = 15
nb_epoch = 3


def make_save_rnn_model(x_train, y_train, x_test, y_test):

    # make model
    model = Sequential()
    # layer word embedding
    model.add(Embedding(max_features, 128))  # 32
    # layer LSTM
    model.add(LSTM(100, dropout=0.3, recurrent_dropout=0.3))
    # layer finish for classify
    model.add(Dense(n_classes, activation='sigmoid'))
    # optimiator
    sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # compile model
    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
    # learn model
    model.fit(x_train, y_train, batch_size, nb_epoch, validation_data=(x_test, y_test), verbose=1)
    # save model to file
    # generate discription in format json
    model_json = model.to_json()
    # write model on file
    json_file = open(model_path, "w")
    json_file.write(model_json)
    json_file.close()
    # save data about weights in format hdf5
    model.save_weights(weights_path)


# make_save_rnn_model(X_train_np, y_train_np, X_test_np, y_test_np)

def load_model(model_path, weights_path):
    # load data about architecture of network
    json_file = open(model_path, "r")
    loaded_model_json = json_file.read()
    json_file.close()
    # make model which based on loaded data
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(weights_path)
    print("Loaded model from disk")
    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return loaded_model


loaded_model = load_model(model_path, weights_path)
print("\n.." * 5)
print(loaded_model.predict(X_test_np[:10])) # np.random.random([1200]))
print("\n.." * 5)
# score = loaded_model.evaluate(X_test_np, y_test_np, verbose=1)
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

