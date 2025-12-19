# STEP 1: Import required libraries

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


# STEP 2: Load Iris dataset

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df['species'] = iris.target


# STEP 3: Separate features and target

X = df.drop('species', axis=1)
# Input features:
# sepal length, sepal width, petal length, petal width

y = df['species']
# Output label (species)


# STEP 4: Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# 80% training data
# 20% testing data


# STEP 5: Create Decision Tree model

model = DecisionTreeClassifier(random_state=42)
# random_state ensures same result every time


# STEP 6: Train the model

model.fit(X_train, y_train)


# STEP 7: Make predictions on test data

y_pred = model.predict(X_test)


# STEP 8: Evaluate model performance

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# STEP 9: Predict species for a NEW input sample

sample = [[5.1, 3.5, 1.4, 0.2]]
# Example flower measurements

prediction = model.predict(sample)

species_map = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}

print("\nPredicted Species:", species_map[prediction[0]])
