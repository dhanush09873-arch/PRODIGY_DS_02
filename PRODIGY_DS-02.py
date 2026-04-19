# Data Cleaning and Exploratory Data Analysis on Titanic Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = sns.load_dataset("titanic")

# Show first 5 rows
print(df.head())

# Show dataset info
print(df.info())

# Check missing values
print(df.isnull().sum())

# Data cleaning
df['age'].fillna(df['age'].mean(), inplace=True)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
df.drop(columns=['deck'], inplace=True)

# Histogram (Age distribution)
plt.hist(df['age'], bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of People")
plt.show()

# Survival count
sns.countplot(x='survived', data=df)
plt.title("Survival Count")
plt.show()
