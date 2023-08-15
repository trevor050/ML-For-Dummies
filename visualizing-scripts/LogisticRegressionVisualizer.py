import matplotlib.pyplot as plt
import pickle 

with open('models/IMDBmodel.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizers/IMDBvectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Assuming `model` is your trained MultinomialNB model
print("loading model")
coef = model.feature_log_prob_[1] - model.feature_log_prob_[0]
print("getting coef")
plt.bar(range(len(coef)), coef)
#turn to image
print("saving image")
plt.savefig('./visualizing-scripts/visualized-scripts/imbdtree.png')
plt.show()