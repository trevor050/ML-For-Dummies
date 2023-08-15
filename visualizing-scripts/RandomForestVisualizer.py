import pickle
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

with open('models/IMDBmodelRF.pkl', 'rb') as f:
    classifier = pickle.load(f)
with open('vectorizers/IMDBvectorizerRF.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

list_of_feature_names = vectorizer.get_feature_names_out().tolist()

tree_in_forest = classifier.estimators_[2]

fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(tree_in_forest, feature_names=list_of_feature_names, filled=True, max_depth=8, ax=ax)
plt.savefig('./visualizing-scripts/visualized-scripts/imbdRFtree.png', dpi=800)
