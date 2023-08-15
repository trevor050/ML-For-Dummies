#open the model IMDBmodel.pkl, and vectorizer called IMDBvectorizer.pkl
import pickle


with open('models/IMDBmodelRF.pkl', 'rb') as f:
    IMDBmodel = pickle.load(f)
with open('vectorizers/IMDBvectorizerRF.pkl', 'rb') as f:
    IMDBvectorizer = pickle.load(f)

#load the model and vectorizer, then add a user input to test the model
def predict_review(review):
    review = IMDBvectorizer.transform([review])
    prediction = IMDBmodel.predict(review)
    return prediction

#test the model
def user_review(review):
    print(predict_review(review))

while True:
    review = input("Enter a review: ")
    user_review(review)