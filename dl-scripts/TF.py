# Welcome to Long Short-Term Memory (LSTM), the sagacious grandparent of the neural network family.
# In a family full of bright minds like BERT, CNN, and RNN, LSTM stands out as the memory keeper, holding the wisdom of the ages.
# Imagine a big family gathering where everyone's sharing memories, secrets, and recipes (and maybe a few white lies).
# The embedding layer is where words get turned into numbers.
# The LSTM layer is where the neural network remembers what it's seen so far.
# The dropout layer is where the neural network forgets some of what it's seen so far. (So it doesn't overfit.)
#Overfitting is when a neural network gets too good at recognizing the training data, and it doesn't do well with new data. Instead of learning, it memorizes.
# The dense layer is where the neural network makes predictions.
# Just like a family reunion, LSTM can be fun, complex, and a little bit overwhelming.
# You've got padding (everybody in their seats!), embedding (connecting on an emotional level), and you're avoiding overfitting (we don't want any family feuds here).

import tensorflow as tf
# Welcome to Long Short-Term Memory (LSTM), the sagacious grandparent of the neural network family.
# In a family full of bright minds like BERT, CNN, and RNN, LSTM stands out as the memory keeper, holding the wisdom of the ages.
# Imagine a big family gathering where everyone's sharing memories, secrets, and recipes (and maybe a few white lies).
# The embedding layer is where words get turned into numbers.
# The LSTM layer is where the neural network remembers what it's seen so far.
# The dropout layer is where the neural network forgets some of what it's seen so far. (So it doesn't overfit.)
#Overfitting is when a neural network gets too good at recognizing the training data, and it doesn't do well with new data. Instead of learning, it memorizes.
# The dense layer is where the neural network makes predictions.
# Just like a family reunion, LSTM can be fun, complex, and a little bit overwhelming.
# You've got padding (everybody in their seats!), embedding (connecting on an emotional level), and you're avoiding overfitting (we don't want any family feuds here).
print("Importing libraries")
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.optimizers import Adam
import pandas as pd

print("Reading data")
data = pd.read_csv('datasets/IMDBCleaned.csv', delimiter=',',  on_bad_lines="skip")
print(f"Data Shape: {data.shape}")

print("Tokenizing and vectorizing")
tokenizer = Tokenizer(num_words=12000, oov_token="<OOV>")
# numwords is the number of words to keep, based on word frequency. oov_token is the token for out of vocabulary words, leave it as <OOV>.
tokenizer.fit_on_texts(data['review'].values)
#no need to vectorize, because TF's tokenizer does that for us. Its like a 2 in 1

print("Padding sequences")
data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})
#this maps the sentiment column to 1 and 0, instead of positive and negative

padded_sequences = pad_sequences(tokenizer.texts_to_sequences(data['review'].values), maxlen=450, padding='post', truncating='post')
# Padding sequences. This makes all the reviews the same length. maxlen is the max length of a review. padding is what to do if the review is shorter than the max length. truncating is what to do if the review is longer than the max length.
print(padded_sequences.shape)

#splitting data
print("Splitting data")
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, data['sentiment'].values, test_size=0.2, random_state=42)


#splititng is the same process as ML models because we are using the same sklearn library. 42 is the answer to life, the universe, and everything, it doesn't matter what number you use. This is just an inside joke that programmers have.




model = tf.keras.Sequential([
    tf.keras.layers.Embedding(12000, 128, input_length=450),
    tf.keras.layers.Conv1D(128, 5, activation='relu'),
    tf.keras.layers.Conv1D(128, 3, activation='relu'), # Additional Conv layer
    tf.keras.layers.MaxPooling1D(pool_size=4),
    tf.keras.layers.LeakyReLU(alpha=0.01),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(2048, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(1024, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512, return_sequences=True)), # Additional LSTM layer
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=False)), # Additional LSTM layer
    tf.keras.layers.ActivityRegularization(l1=1e-5, l2=1e-4), # Regularization
    tf.keras.layers.Dropout(0.5), # Increased Dropout
    tf.keras.layers.Dense(1024, activation='relu'), # Increased neurons
    tf.keras.layers.LeakyReLU(alpha=0.01),
    tf.keras.layers.Dense(512, activation='relu'), # Increased neurons
    tf.keras.layers.Dropout(0.5), # Increased Dropout
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])



optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# And fit it just like you would before
history = model.fit(X_train, y_train, epochs=8, batch_size=32)

# Binary crossentropy is the loss function for binary classification problems, in simple terms its best for 0 or 1 outputs. Adam is the optimizer, and accuracy is the metric we want to track.)
print("saving tokenizer")
import pickle
print("Training model")

history = model.fit(X_train, y_train, epochs=10, batch_size=64)
# Epochs is the number of times the model will see the training data. Batch size is the number of training examples per batch. The model will see 32 training examples at a time, and it will see the entire training data 10 times.



print("Evaluating model")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Loss: {loss}, Accuracy: {accuracy}")

print("Saving model")
model.save("bestmodeleva.h5")
model.save("dl-models/neoIMDBLTST.h5")
with open('train_history.pkl', 'wb') as file:
    pickle.dump(history.history, file)
print("saving tokenizer")
with open('neoimdbtokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


#two different columns. review, and sentiment

