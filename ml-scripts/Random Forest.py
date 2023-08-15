# Welcome to Random Forest! This beast takes a crowd of IMDb reviews and lets a forest of decision trees wrestle with them.
# Unlike the simpleton NMB that treats words like lone wolves, Random Forest forms a council of wise trees.
# Each tree looks at a random subset of the reviews and words, so they don't just copy each other's homework.
# They vote on whether a review is positive or negative, just like reality TV judges but less dramatic.
# The result? A smarter prediction that understands words can be buddies.

# It's a bit slower than NMB (like comparing a turtle to a rabbit), but more reliable. A bit like trading a skateboard for a bike.
# You want a wise council of trees? You've got it! You want a model that doesn't assume words hate each other? You've got that too!
# It's the jungle party of machine learning! And guess what, you're the Tarzan here! Swing through those decisions! 🌲🌳🦍




print("Importing Libraries")
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

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


rf_model = RandomForestClassifier(n_estimators=1000, n_jobs=-1)





print("Training the model...")
rf_model.fit(X_train_vectorized, y_train)
rf_accuracy = rf_model.score(X_test_vectorized, y_test)

print(f"Accuracy: {rf_accuracy}")

# Heres an input area for you to test the ai
your_review = input("Enter a review: ")
#test it against the model
your_review_vectorized = vectorizer.transform([your_review])
prediction = rf_model.predict(your_review_vectorized)
print(prediction)



# Define the path to save the models
ModelsPath = './models'
VectorizersPath = './vectorizers'

# Save the model
with open(os.path.join(ModelsPath, 'IMDBModelRF.pkl'), 'wb') as model_file:
    pickle.dump(rf_model, model_file)

# Save the TF-IDF vectorizer
with open(os.path.join(VectorizersPath, 'IMDBVectorizerRF.pkl'), 'wb') as vector_file:
    pickle.dump(vectorizer, vector_file)