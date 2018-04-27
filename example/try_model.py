import pandas as pn
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split



data = pn.read_csv('/Users/andreybezumennui/PycharmProjects/nlp_sanbox_/nlp-sandbox/ner_multi-label_classify/datasets/all_39k_22c_2201.csv')


# количество эпох\итераций для обучения
epochs = 10
num_words = 8104


X_train_counts = data['Data'].tolist()
Y_train_counts = data['Intent'].tolist()

tokenizer = CountVectorizer().fit_transform(X_train_counts)
total_words = len(tokenizer.shape)

X_train, X_test, y_train, y_test = train_test_split(X_train_counts, Y_train_counts, test_size=0.3)


