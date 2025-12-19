import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()

# Iris dataset is now loaded into a variable called 'iris'
# It contains:
# - data (features)
# - target (species)
# - feature names

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# 1 Histogram
# To understand how values are distributed for each feature
plt.figure(figsize=(8, 6))
df[['sepal length (cm)',
    'sepal width (cm)',
    'petal length (cm)',
    'petal width (cm)']].hist()
plt.suptitle("Feature Distribution")
plt.show()
plt.close()