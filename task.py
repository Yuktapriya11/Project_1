# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv(r"C:\Users\hp\Downloads\archive (1)\StudentPerformanceFactors.csv")  # <-- Replace with your filename if different

# Display first few rows
print("First 5 rows:")
print(df.head())

# Data summary
print("\nDataset Info:")
print(df.info())

print("\nMissing values per column:")
print(df.isnull().sum())

# Drop duplicates
df = df.drop_duplicates()
print(f"\nShape after removing duplicates: {df.shape}")

# Descriptive statistics
print("\nSummary Statistics:")
print(df.describe())

# Gender-wise average Exam Score
gender_avg = df.groupby('Gender')['Exam_Score'].mean()
print("\nAverage Exam Score by Gender:")
print(gender_avg)

# Students scoring above 85
high_scores = df[df['Exam_Score'] > 85]
print(f"\nNumber of students scoring above 85: {high_scores.shape[0]}")

# Correlation matrix
corr_matrix = df.corr(numeric_only=True)
print("\nCorrelation Matrix:")
print(corr_matrix['Exam_Score'].sort_values(ascending=False))

# Visualization 1: Exam Score Distribution
plt.figure(figsize=(8, 5))
plt.hist(df['Exam_Score'], bins=10, color='mediumseagreen', edgecolor='black')
plt.title('Distribution of Exam Scores')
plt.xlabel('Exam Score')
plt.ylabel('Number of Students')
plt.grid(True)
plt.show()

# Visualization 2: Study Hours vs Exam Score
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='Hours_Studied', y='Exam_Score', hue='Gender')
plt.title('Hours Studied vs Exam Score')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.grid(True)
plt.show()

# Visualization 3: Average Exam Score by Gender
plt.figure(figsize=(6, 4))
gender_avg.plot(kind='bar', color=['skyblue', 'lightpink'])
plt.title('Average Exam Score by Gender')
plt.ylabel('Average Exam Score')
plt.xticks(rotation=0)
plt.grid(True)
plt.show()

# Visualization 4: Heatmap of Correlation Matrix
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()
