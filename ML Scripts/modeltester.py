#open the model IMDBmodel.pkl, and vectorizer called IMDBvectorizer.pkl
import pickle


with open('Models/IMDBmodel.pkl', 'rb') as f:
    IMDBmodel = pickle.load(f)
with open('vectorizers/IMDBvectorizer.pkl', 'rb') as f:
    IMDBvectorizer = pickle.load(f)

#load the model and vectorizer, then add a user input to test the model
def predict_review(review):
    review = IMDBvectorizer.transform([review])
    prediction = IMDBmodel.predict(review)
    return prediction

#test the model
review = 'This movie is great'
prediction = predict_review(review)
print(prediction)