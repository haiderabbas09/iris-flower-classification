from sklearn import tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Step 1: Collect Data
# Features: [weight (grams), color (0 = red, 1 = orange)]
features = [
    [150, 0],  # Apple
    [170, 0],  # Apple
    [130, 1],  # Orange
    [120, 1],  # Orange
]
labels = ["apple", "apple", "orange", "orange"]

# Step 2: Train a model
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

# Step 3: Visualize the decision tree
plt.figure(figsize=(9, 2))
plot_tree(clf, feature_names=["height", "color"], class_names=["apple", "orange"], filled=True)
plt.show()

# Step 4: Predict!
print(clf.predict([[300, 0]]))  # Heavy and red-ish? Probably an apple!
print(clf.predict([[115, 1]]))  # Light and orange? Probably an o
print(f"this is a {clf.predict([[115, 0]])[0]}")  # Light and orange? Probably an orange!