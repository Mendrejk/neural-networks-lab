import csv

from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dense, LSTM
from keras.utils import pad_sequences
import tensorflow as tf
from keras import backend

config = tf.compat.v1.ConfigProto(device_count={'GPU': 0, 'CPU': 2})
config.gpu_options.per_process_gpu_memory_fraction = 0.75
sess = tf.compat.v1.Session(config=config)
backend.set_session(sess)

# Set the maximum number of words to include in the review
max_features = 10000
# Cut off the text after this number of words (among the max_features most common words)
max_len = 500


def create_model(embedding_dim, mask_zero):
    new_model = Sequential()
    new_model.add(Embedding(max_features, embedding_dim, mask_zero=mask_zero, input_length=max_len))
    new_model.add(SimpleRNN(100))
    # new_model.add(LSTM(100)) # alternative to SimpleRNN
    new_model.add(Dense(1, activation="sigmoid"))
    new_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    return new_model


embedding_dims = [50, 100, 150, 200, 350, 500]
mask_zero_values = [True, False]

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

# Study the hyperparameters
for mask_zero in mask_zero_values:
    # Create a CSV file to save the results
    with open("results_{}.csv".format("mask_zero" if mask_zero else "no_mask_zero"), "w") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Embedding dimension", "Accuracy"])

        for embedding_dim in embedding_dims:
            # Train the model and evaluate its accuracy
            model = create_model(embedding_dim, mask_zero)
            model.fit(x_train, y_train, batch_size=200, epochs=3, validation_split=0.2)
            _, accuracy = model.evaluate(x_test, y_test, batch_size=100)

            # Print the results to the console
            print("Mask zero: {} - Embedding dim: {} - Accuracy: {:.3f}".format(mask_zero, embedding_dim, accuracy))

            # Save the results to the CSV file
            csvwriter.writerow([embedding_dim, accuracy])

# model = KerasClassifier(build_fn=create_model, epochs=3, batch_size=100)
# param_grid = {
#     "embedding_dim": [50, 100, 150, 200, 350, 500],
#     "mask_zero": [True, False]
# }
# grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, error_score='raise')
# grid_search.fit(x_train, y_train)

# print("Best hyperparameters:", grid_search.best_params_)

# results = grid_search.cv_results_
# for mean_score, params in zip(results["mean_test_score"], results["params"]):
#     print(mean_score, params)
#
# with open("hyperparameter_results.csv", "w") as csvfile:
#     writer = csv.writer(csvfile)
#
#     # Header
#     writer.writerow(["accuracy", "embedding_dim", "mask_zero"])
#
#     for mean_score, params in zip(results["mean_test_score"], results["params"]):
#         writer.writerow([mean_score, params["embedding_dim"], params["mask_zero"]])
