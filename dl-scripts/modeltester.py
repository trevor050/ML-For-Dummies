import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
# Load the model
model = tf.keras.models.load_model('IMDBLTST2.h5')

#load the tokenizer
with open('imdbtokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Get user input



# Make a prediction
def prediction(test_input):
    prediction = model.predict(test_input)
    if prediction >= 0.5:
        print("Positive review")
    else:
        print("Negative review")
    print(prediction)

while True:
    test_input = input("Enter a review: ")

    # Tokenize the input
    tokenizer.fit_on_texts([test_input])
    test_input = tokenizer.texts_to_sequences([test_input])

    # Pad the input
    test_input = tf.keras.preprocessing.sequence.pad_sequences(test_input, maxlen=200)
    prediction(test_input)