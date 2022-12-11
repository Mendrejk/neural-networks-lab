from keras.datasets import imdb
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dense, Masking
import tensorflow as tf
from keras import backend

config = tf.compat.v1.ConfigProto(device_count={'GPU': 1, 'CPU': 4})
config.gpu_options.per_process_gpu_memory_fraction = 0.75
sess = tf.compat.v1.Session(config=config)
backend.set_session(sess)

# Set the maximum number of words to include in the review
max_features = 10000
# Cut off the text after this number of words (among the max_features most common words)
max_len = 500

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

model = Sequential()
# 1000 dimensions and masking
model.add(Embedding(max_features, 1000, mask_zero=True))
model.add(SimpleRNN(100))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=100, epochs=10, validation_split=0.2)