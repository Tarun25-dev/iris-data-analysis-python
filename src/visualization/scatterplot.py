import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
import seaborn as sns

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
df['species'] = iris.target


# 3 Scatter Plot
# To understand relationship between petal length and petal width
plt.figure(figsize=(6, 4))
sns.scatterplot(
    x="petal length (cm)",
    y="petal width (cm)",
    hue="species",
    data=df
)
plt.title("Petal Length vs Petal Width")
plt.show()
plt.close()
