
# Accident Data Visualization

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


sns.set(style="whitegrid", palette="cool")
df = pd.read_csv('/home/aura/Documents/2.2/datascience/accidents.csv')
plt.figure(figsize=(8,5))
sns.countplot(data=df, x='Accident_severity', color='royalblue')
plt.title('Distribution of Accident Severity')
plt.xlabel('Accident Severity Level')
plt.ylabel('Number of Accidents')
plt.show()



plt.figure(figsize=(10,6))
sns.boxplot(data=df, x='Age_band_of_driver', y='Accident_severity', palette='Blues')
plt.title('Accident Severity vs Age Band of Driver')
plt.xlabel('Age Band of Driver')
plt.ylabel('Accident Severity')
plt.xticks(rotation=45)
plt.show()

# drive exper
plt.figure(figsize=(10,6))
sns.boxplot(data=df, x='Driving_experience', y='Accident_severity', palette='Greens')
plt.title('Accident Severity vs Driving Experience')
plt.xlabel('Driving Experience')
plt.ylabel('Accident Severity')
plt.xticks(rotation=45)
plt.show()

# road surface
plt.figure(figsize=(10,6))
sns.barplot(
    data=df.groupby('Road_surface_type')['Accident_severity'].mean().reset_index(),
    x='Road_surface_type', y='Accident_severity', palette='mako'
)
plt.title('Average Accident Severity by Road Surface Type')
plt.xlabel('Road Surface Type')
plt.ylabel('Average Severity')
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(12,8))
sns.heatmap(X.corr(), cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap (Encoded Features)')
plt.show()
