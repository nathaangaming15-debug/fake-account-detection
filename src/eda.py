import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('reports', exist_ok=True)

df = pd.read_csv('data/train.csv')

print(df.head())
print(df.info())
print(df.isnull().sum())

# Class balance
sns.countplot(x='fake', data=df)
plt.title('Fake vs Real Account Distribution')
plt.savefig('reports/class_distribution.png')
plt.close()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.savefig('reports/correlation_heatmap.png')
plt.close()

print("EDA complete. Charts saved to reports/")
