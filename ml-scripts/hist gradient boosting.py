# Welcome to Gradient Boosting, the random forest's older, wiser, and much cooler brother.
# It's a bit like a relay race, where each decision tree (or runner) corrects the errors of the previous one.
# 'n_estimators' is the number of trees (or racers) in the relay. More can be better, but don't go overboard.
# 'learning_rate' is like the training intensity, determining how much each tree adjusts. A delicate balance is key.
# 'max_depth'? Think of it as how much each tree can grow. More might lead to overfitting, like overtraining a racer.

# It's more time-consuming than some other models (like making a homemade meal vs. fast food), but often tastier.
# Want a model that learns from its mistakes? Check! Want to tweak and tune like a master chef? You've got it!
# Gradient Boosting is all about slow-cooking perfection. So grab your apron, and let's cook up some accuracy! 


print("Importing Libraries")
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import HistGradientBoostingClassifier

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






# Replace NaN values with the mean of the column

print("Training the model...")

gb_model = HistGradientBoostingClassifier(learning_rate=0.1, max_depth=10)
#densify data
X_train_vectorized = X_train_vectorized.todense()
X_test_vectorized = X_test_vectorized.todense()
#Please convert to a numpy array with np.asarray
X_train_vectorized = np.asarray(X_train_vectorized)
X_test_vectorized = np.asarray(X_test_vectorized)

gb_model.fit(X_train_vectorized, y_train)
gb_accuracy = gb_model.score(X_test_vectorized, y_test)

print(f"Accuracy: {gb_accuracy}")

# Heres an input area for you to test the ai
your_review = input("Enter a review: ")
#test it against the model
your_review_vectorized = vectorizer.transform([your_review])
prediction = gb_model.predict(your_review_vectorized)
print(prediction)



# Define the path to save the models
ModelsPath = './models'
VectorizersPath = './vectorizers'

# Save the model
with open(os.path.join(ModelsPath, 'IMDBModelHGB.pkl'), 'wb') as model_file:
    pickle.dump(gb_model, model_file)

# Save the TF-IDF vectorizer
with open(os.path.join(VectorizersPath, 'IMDBVectorizerHGB.pkl'), 'wb') as vector_file:
    pickle.dump(vectorizer, vector_file)