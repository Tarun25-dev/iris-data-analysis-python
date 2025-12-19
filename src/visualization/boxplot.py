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

# 2 Boxplot
# To compare petal length across species
plt.figure(figsize=(6, 4))
sns.boxplot(x="species", y="petal length (cm)", data=df)
plt.title("Petal Length by Species")
plt.show()
plt.close()
