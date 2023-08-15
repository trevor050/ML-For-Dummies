#this takes a list of a shit ton of imdb reviews that have postive or negative sentiment and then using a simple model known as NMB that reads each word individually and then figures out what output its most similar too.
#the issue with this is that its not too smart, its honestly really dumb. It assumes words dont work together. It has no complexity than surface level.
#Its really fucking fast though (For this massive dataset of over 50k long reviews it trains in a second)


print("Importing Libraries")
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import numpy as np




# Read the CSV file with on_bad_lines set to skip errors
print("Reading data")
data = pd.read_csv('datasets/IMDBCleaned.csv', delimiter=',',  on_bad_lines="skip")
print(f"Data Shape: {data.shape}")


#getting a column from called review, and sentiment. the two we need
review = data['review']

sentiments = data['sentiment']

# Split the dataset into training and test sets
#X is the review, y is the sentiment
#we are splitting 80% of x to teach our ai with. then 20% to test it. Same with y (test size is how much you test with, go for 0.2). Random state is just a random number generator's seed. 42 is just the meaning to life (just an inside joke doesnt matter what it is)
print("Splitting data")
X_train, X_test, y_train, y_test = train_test_split(review, sentiments, test_size=0.2, random_state=42)



# Create a TF-IDF vectorizer (turns text into numbers, floats to be exact)
print("Vectorizing data")
vectorizer = TfidfVectorizer()


# Fit and transform the training data (vectorize it)
X_train_vectorized = vectorizer.fit_transform(X_train)


# Transform the test data
X_test_vectorized = vectorizer.transform(X_test)


#you only need to vectorize x, not y. y is just the sentiment. (even if y is text you still dont need to vectorize it). Reason: its learning


#TDIF is good for text classification because it not only turns text into numbers that the computer can understand, but it also gives more weight to words that appear more often in a particular document, and less weight to words that appear in many documents. its a 2 in 1
# Create a Multinomial Naive Bayes model
model = MultinomialNB()

#In simple terms, a Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature. For example, a fruit may be considered to be an apple if it is red, round, and about 3 inches in diameter. Even if these features depend on each other or upon the existence of the other features, a naive Bayes classifier would consider all of these properties to independently contribute to the probability that this fruit is an apple.

# A quick reminder about the Naive Bayes model:
# It's like a word detective. We're looking at words in tweets and counting
# how many happy and sad words there are. If there's more sad than happy,
# it guesses the tweet is negative. If more happy, it's positive. It's not
# super smart, assumes words don't play together, but surprisingly works!
# Keep unraveling those text mysteries! 🔍🕵️‍♂️😄
# Train the model with the training data


# Replace NaN values with the mean of the column

print("Training the model...")
model.fit(X_train_vectorized, y_train)

y_pred = model.predict(X_test_vectorized)
print("Accuracy: ", accuracy_score(y_test, y_pred))

# Heres an input area for you to test the ai
your_review = input("Enter a review: ")
#test it against the model
your_review_vectorized = vectorizer.transform([your_review])
prediction = model.predict(your_review_vectorized)
print(prediction)



# Define the path to save the models
ModelsPath = './models'
VectorizersPath = './vectorizers'

# Save the model
with open(os.path.join(ModelsPath, 'IMDBModel.pkl'), 'wb') as model_file:
    pickle.dump(model, model_file)

# Save the TF-IDF vectorizer
with open(os.path.join(VectorizersPath, 'IMDBVectorizer.pkl'), 'wb') as vector_file:
    pickle.dump(vectorizer, vector_file)