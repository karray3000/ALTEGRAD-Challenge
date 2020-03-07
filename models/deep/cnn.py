import codecs
from os import path
import os
import re
import wget
from sklearn.base import BaseEstimator
from keras.preprocessing.text import Tokenizer
from sklearn import preprocessing


from gensim.models import KeyedVectors
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Concatenate, Dense, MaxPooling1D, LSTM
from tensorflow.keras.initializers import Constant
from keras.preprocessing import sequence

labels = ['business/finance', 'education/research', 'entertainment', 'health/medical',
          'news/press', 'politics/government/law', 'sports', 'tech/science']

d = 200
max_size = 1000
max_words = 150000
nb_branches = 2
nb_filters = 50
filter_sizes = [3, 4]
drop_rate = 0.3
batch_size = 64
nb_epochs = 3
my_optimizer = 'adam'

def get_all_texts():
    text_path = 'data/text/'
    filenames = os.listdir(text_path)
    all_texts = {}
    for filename in filenames:
        try:
            with codecs.open(path.join(text_path, filename), encoding='utf-8') as f:
                all_texts[filename] = f.read().replace("\n", "").lower()
        except:
            with codecs.open(path.join(text_path, filename), encoding='latin-1') as f:
                all_texts[filename] = f.read().replace("\n", "").lower()
    return all_texts


all_texts = get_all_texts()
pattern = re.compile('[\W_]+')
all_texts_cleaned = {host: pattern.sub(' ', text) for host, text in all_texts.items()}
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(all_texts_cleaned.values())

try:
    wget.download("http://embeddings.net/frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin",
                  'frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin')
except:
    pass

embeds = KeyedVectors.load_word2vec_format('frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin', binary=True)
word_index = tokenizer.word_index
embedding_matrix = np.zeros((max_words, d))

for word, i in word_index.items():
    if i >= max_words:
        continue
    try:
        embedding_vector = embeds[word]
        embedding_matrix[i] = embedding_vector
    except KeyError:
        pass

def cnn_branch(n_filters, k_size, d_rate, my_input):
    return Dropout(d_rate)(GlobalMaxPooling1D()(Conv1D(filters=n_filters,
                                                       kernel_size=k_size,
                                                       activation='relu')(my_input)))

def smooth_labels(labels, factor=0.1):
    labels *= (1 - factor)
    labels += (factor / labels.shape[1])
    return labels



class CNNModel(BaseEstimator):
    def __init__(self):
        embedding_layer = Embedding(max_words,
                                    d,
                                    embeddings_initializer=Constant(embedding_matrix),
                                    input_length=max_size,
                                    trainable=True)
        doc_ints = Input(shape=(None,))
        doc_wv = embedding_layer(doc_ints)
        doc_wv_dr = Dropout(drop_rate)(doc_wv)

        branch_outputs = []

        for idx in range(nb_branches):
            branch_outputs.append(cnn_branch(nb_filters, filter_sizes[idx], drop_rate, doc_wv_dr))

        concat = Concatenate()(branch_outputs)
        preds = Dense(units=8, activation='softmax')(concat)
        self.model = Model(doc_ints, preds)

        self.model.compile(loss='categorical_crossentropy', optimizer=my_optimizer)
        self.le = preprocessing.LabelBinarizer()
        self.le.fit([u for u in labels])

    def fit(self, X_train, y_train):
        train_data = sequence.pad_sequences(tokenizer.texts_to_sequences([pattern.sub(' ', u[1]) for u in X_train]),
                                            maxlen=max_size)

        y_tr = self.le.transform([u for u in y_train])
        y_tr = smooth_labels(np.array(y_tr, dtype='float64'))
        res = self.model.fit(np.array(train_data), np.array(y_tr), batch_size=16, epochs=15)
        return res

    def predict_proba(self, X_test):
        test_data = sequence.pad_sequences(tokenizer.texts_to_sequences([pattern.sub(' ', u[1]) for u in X_test]),
                                           maxlen=max_size)
        return self.model.predict(test_data)
