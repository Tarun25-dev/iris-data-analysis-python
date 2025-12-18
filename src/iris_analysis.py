# STEP 1: Import required libraries

import numpy as np
# NumPy is used for numerical calculations like mean, max, etc.

import pandas as pd
# Pandas is used to work with data in table (row & column) format

from sklearn.datasets import load_iris
# sklearn provides the Iris dataset directly


# STEP 2: Load the Iris dataset

iris = load_iris()
# Iris dataset is now loaded into a variable called 'iris'
# It contains:
# - data (features)
# - target (species)
# - feature names


# STEP 3: Convert NumPy array to Pandas DataFrame

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)
# iris.data is a NumPy array with 150 rows and 4 columns
# We convert it into a Pandas DataFrame for easy analysis


# STEP 4: Add target column (species)

df['species'] = iris.target
# Target column contains values:
# 0 -> Setosa
# 1 -> Versicolor
# 2 -> Virginica


# STEP 5: Initial data inspection

print("First 5 rows of the dataset:")
print(df.head())
# Shows first 5 rows to verify data is loaded correctly

print("\nDataset information:")
print(df.info())
# Shows column names, data types, and confirms no missing values

print("\nStatistical summary:")
print(df.describe())
# Gives statistics like mean, min, max, etc.


# STEP 6: NumPy-based statistical operations

mean_sepal_length = np.mean(df['sepal length (cm)'])
# Calculates average sepal length using NumPy

max_petal_length = np.max(df['petal length (cm)'])
# Finds maximum petal length using NumPy

print("\nMean Sepal Length:", mean_sepal_length)
print("Max Petal Length:", max_petal_length)


# STEP 7: Species-wise analysis using Pandas groupby

species_mean = df.groupby('species').mean()
# Groups data by species and calculates mean for each group

print("\nSpecies-wise mean values:")
print(species_mean)


# STEP 8: Logical insights (simple observations)

print("\nInsights:")
print("- Setosa species has smaller petal dimensions")
print("- Virginica species has larger petal dimensions")
print("- Feature values help differentiate species clearly")


# STEP 9: Project conclusion

print("\nConclusion:")
print(
    "This project shows how a standard dataset can be analyzed "
    "using NumPy for calculations and Pandas for data analysis."
)