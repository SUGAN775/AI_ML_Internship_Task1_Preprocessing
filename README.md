# 1. Necessary Libraries Import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from scipy import stats

# 2. Dataset Loading and Initial Exploration
print("--- 1. Data Loading and Initial Exploration ---")
# Replace 'titanic.csv' with your dataset path
df = pd.read_csv('titanic.csv') 
print(df.info())
print("\nMissing values count:\n", df.isnull().sum())

# ----------------------------------------------------------------------
# 3. Handling Missing Values (Null Imputation)
print("\n--- 2. Handling Missing Values ---")

# a) Numerical Feature Imputation (Age): Filling with Mean
# We use the mean for imputation
mean_age = df['Age'].mean()
df['Age'].fillna(mean_age, inplace=True)
print(f"Age column filled with Mean: {mean_age:.2f}")

# b) Categorical Feature Imputation (Embarked): Filling with Mode
mode_embarked = df['Embarked'].mode()[0]
df['Embarked'].fillna(mode_embarked, inplace=True)

# c) High Null Count Column Removal (Cabin)
# Dropping 'Cabin' due to excessive missing data
df.drop('Cabin', axis=1, inplace=True) 
print("\nCabin column dropped. Remaining nulls:\n", df.isnull().sum())

# ----------------------------------------------------------------------
# 4. Outlier Detection and Handling
print("\n--- 3. Outlier Detection (Age Example) ---")

# Visualization using Boxplot
plt.figure(figsize=(6, 4))
sns.boxplot(y=df['Age'])
plt.title('Boxplot of Age before Outlier Handling')
plt.show() # 

# Outlier Capping using IQR Method
Q1 = df['Age'].quantile(0.25)
Q3 = df['Age'].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR
lower_bound = Q1 - 1.5 * IQR

# Applying Capping (replacing outliers with boundary values)
df['Age'] = np.where(df['Age'] > upper_bound, upper_bound, df['Age'])
df['Age'] = np.where(df['Age'] < lower_bound, lower_bound, df['Age'])
print("Outliers in Age capped using IQR method.")

# ----------------------------------------------------------------------
# 5. Categorical Encoding
print("\n--- 4. Categorical Encoding ---")

# a) One-Hot Encoding (for Nominal Data - e.g., Embarked)
# Using get_dummies to create binary columns
df = pd.get_dummies(df, columns=['Embarked'], prefix='Embarked', drop_first=True)
print("Embarked One-Hot Encoded.")

# b) Label Encoding (for Binary/Ordinal Data - e.g., Sex)
le = LabelEncoder()
df['Sex_Encoded'] = le.fit_transform(df['Sex'])
df.drop('Sex', axis=1, inplace=True)
print("Sex Label Encoded.")

# ----------------------------------------------------------------------
# 6. Feature Scaling
print("\n--- 5. Feature Scaling (Fare Example) ---")

# Selecting the feature to scale
fare = df[['Fare']]

# a) Normalization (MinMaxScaler) - scales to [0, 1]
scaler_norm = MinMaxScaler()
df['Fare_Normalized'] = scaler_norm.fit_transform(fare)
print("Fare column Normalized (scaled to 0-1).")

# b) Standardization (StandardScaler) - scales to mean=0, std=1
scaler_std = StandardScaler()
df['Fare_Standardized'] = scaler_std.fit_transform(fare)
print("Fare column Standardized (scaled to mean=0, std=1).")

# ----------------------------------------------------------------------
# Final Output
print("\nFinal Data Head after Preprocessing:\n", df.head())
