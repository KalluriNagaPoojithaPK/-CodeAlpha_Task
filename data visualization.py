import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import StringIO

# Function to load the Titanic dataset
def load_data(url):
    response = requests.get(url)
    return pd.read_csv(StringIO(response.text))

# Load the Titanic dataset
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
data = load_data(url)

# Data Cleaning
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
data.drop('Cabin', axis=1, inplace=True)

# Convert categorical variables to numeric
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Embarked'] = data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Set the visual style
sns.set(style="whitegrid")

# Create a figure for multiple subplots
fig, axes = plt.subplots(3, 2, figsize=(15, 15))

# 1. Age Distribution
sns.histplot(data['Age'], bins=30, kde=True, ax=axes[0, 0], color='skyblue')
axes[0, 0].set_title('Age Distribution', fontsize=16)
axes[0, 0].set_xlabel('Age', fontsize=14)
axes[0, 0].set_ylabel('Frequency', fontsize=14)

# 2. Survival Rate by Gender
sns.barplot(x='Sex', y='Survived', data=data, ax=axes[0, 1], palette='pastel')
axes[0, 1].set_title('Survival Rate by Gender', fontsize=16)
axes[0, 1].set_xlabel('Gender (0 = Male, 1 = Female)', fontsize=14)
axes[0, 1].set_ylabel('Survival Rate', fontsize=14)

# 3. Survival Rate by Passenger Class
sns.barplot(x='Pclass', y='Survived', data=data, ax=axes[1, 0], palette='pastel')
axes[1, 0].set_title('Survival Rate by Passenger Class', fontsize=16)
axes[1, 0].set_xlabel('Passenger Class', fontsize=14)
axes[1, 0].set_ylabel('Survival Rate', fontsize=14)

# 4. Family Size Analysis
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
sns.barplot(x='FamilySize', y='Survived', data=data, ax=axes[1, 1], palette='pastel')
axes[1, 1].set_title('Survival Rate by Family Size', fontsize=16)
axes[1, 1].set_xlabel('Family Size', fontsize=14)
axes[1, 1].set_ylabel('Survival Rate', fontsize=14)

# 5. Fare Distribution by Survival
sns.boxplot(x='Survived', y='Fare', data=data, ax=axes[2, 0], palette='pastel')
axes[2, 0].set_title('Fare Distribution by Survival', fontsize=16)
axes[2, 0].set_xlabel('Survived', fontsize=14)
axes[2, 0].set_ylabel('Fare', fontsize=14)

# 6. Correlation Heatmap
plt.subplot(3, 2, 6)
correlation_matrix = data.select_dtypes(include=['int64', 'float64']).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
plt.title('Feature Correlation Heatmap', fontsize=16)

# Adjust layout
plt.tight_layout()
plt.show()

# Save the visualizations to files
fig.savefig('titanic_visualizations.png')
