import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load and Explore the Dataset (Task 1)
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Display the first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Explore the structure of the dataset
print("\nDataset Info:")
print(df.info())

print("\nMissing values in each column:")
print(df.isnull().sum())

# Clean the dataset (no missing values in Iris dataset)

# Basic Data Analysis (Task 2)

# Basic statistics
print("\nBasic Statistics:")
print(df.describe())

# Group by species and compute mean
print("\nMean values by species:")
print(df.groupby('species').mean())

# Mean Sepal Length by Species
mean_sepal_length = df.groupby('species')['sepal length (cm)'].mean()
print("\nMean Sepal Length by Species:")
print(mean_sepal_length)

# Data Visualization (Task 3)

# Setup for visualizations
sns.set(style="whitegrid")
species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
df['species_name'] = df['species'].map(species_map)

# 1. Line Chart: Sepal and Petal Length over Index
plt.figure(figsize=(10, 4))
plt.plot(df.index, df['sepal length (cm)'], label='Sepal Length')
plt.plot(df.index, df['petal length (cm)'], label='Petal Length')
plt.title("Line Chart: Sepal and Petal Lengths Over Index")
plt.xlabel("Index")
plt.ylabel("Length (cm)")
plt.legend()
plt.show()

# 2. Bar Chart: Average Petal Length by Species
plt.figure(figsize=(6, 4))
sns.barplot(x='species_name', y='petal length (cm)', data=df)
plt.title("Bar Chart: Average Petal Length by Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length (cm)")
plt.show()

# 3. Histogram: Sepal Width Distribution
plt.figure(figsize=(6, 4))
sns.histplot(df['sepal width (cm)'], bins=20, kde=True, color='skyblue')
plt.title("Histogram: Sepal Width Distribution")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.show()

# 4. Scatter Plot: Sepal Length vs Petal Length
plt.figure(figsize=(6, 4))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species_name', data=df)
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title='Species')
plt.show()
