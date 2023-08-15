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

padded_sequences = pad_sequences(tokenizer.texts_to_sequences(data['review'].values), maxlen=200, padding='post', truncating='post')
# Padding sequences. This makes all the reviews the same length. maxlen is the max length of a review. padding is what to do if the review is shorter than the max length. truncating is what to do if the review is longer than the max length.
print(padded_sequences.shape)

#splitting data
print("Splitting data")
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, data['sentiment'].values, test_size=0.2, random_state=42)


#splititng is the same process as ML models because we are using the same sklearn library. 42 is the answer to life, the universe, and everything, it doesn't matter what number you use. This is just an inside joke that programmers have.

print("Building model")
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(9000, 32, input_length=200), # Word fingerprint. 9k vocab size (he max amount words the model can remember), 64 dimensions (the size of the word fingerprint, larger is better but more computationally expensive), 400 words per review (the max length of a review).
    tf.keras.layers.Conv1D(32, 5, activation='relu'), # Convolutional layer. 32 filters (the number of different fingerprints the model will learn), 5 words per fingerprint (the size of the fingerprint), relu activation function (the function that decides whether a neuron fires or not
    tf.keras.layers.MaxPooling1D(5), # Pooling layer. 5 words per fingerprint (the size of the fingerprint)
    tf.keras.layers.GRU(32, dropout=0.2, recurrent_dropout=0.2), # LSTM with dropout. Dropout is a regularization technique that helps prevent overfitting. It randomly drops some of the neurons while training, so the neural network doesn't get too good at recognizing the training data. Overfitting is when a neural network gets too good at recognizing the training data, and it doesn't do well with new data. Instead of learning, it memorizes.
    tf.keras.layers.Dropout(0.5), # Extra dropout layer, just to be sure
    tf.keras.layers.Dense(1, activation='sigmoid') # Decision-making. The output layer. 1 neuron, because we're doing binary classification. Sigmoid activation function, because we want a probability between 0 and 1.
])

print("Compiling model")
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Binary crossentropy is the loss function for binary classification problems, in simple terms its best for 0 or 1 outputs. Adam is the optimizer, and accuracy is the metric we want to track.)
print("saving tokenizer")
import pickle
print("Training model")

model.fit(X_train, y_train, epochs=3, batch_size=32)
# Epochs is the number of times the model will see the training data. Batch size is the number of training examples per batch. The model will see 32 training examples at a time, and it will see the entire training data 10 times.



print("Evaluating model")
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Loss: {loss}, Accuracy: {accuracy}")
with open('imdbtokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Saving model")
model.save("bestmodeleva.h5")
model.save("dl-models/IMDBLTST.h5")

print("saving tokenizer")

#two different columns. review, and sentiment

