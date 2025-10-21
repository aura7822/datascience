📊 Data Science with Python – School Assignments & CAT Tasks

 📌 Orientation

This repository contains all my school assignments and CAT (Continuous Assessment Test) tasks related to Data Science and Python programming. It is designed as a learning hub where I apply theoretical concepts to practical problems using Python and its data science ecosystem.
---

### 🧠 What is Data Science?

Data Science is an interdisciplinary field that uses statistics, mathematics, and computer science to extract meaningful insights from data. It involves:

Data Collection – gathering raw information from different sources (e.g., CSV files, databases, APIs, web scraping).

Data Cleaning – preprocessing and handling missing, duplicate, or noisy data.

Exploratory Data Analysis (EDA) – using visualization and statistics to understand patterns.

Model Building – applying machine learning techniques to predict or classify outcomes.

Communication – presenting findings clearly through reports, visualizations, and dashboards.
◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘

### 🐍 Why Python for Data Science?

Python is the most popular programming language for data science because it is:

Simple & Readable – great for beginners and professionals alike.
◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘

### Rich Ecosystem – libraries such as:

numpy → numerical computing

pandas → data manipulation

matplotlib & seaborn → visualization

scikit-learn → machine learning

seaborn - aesthetic visualisation

matplotlib - basic plotting

Open-source & Community-driven – widely supported with tons of resources.
◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘◘

### 📂 Repository Structure :

The repository is organized into folders:

🢣 Python libraries

🢣 Data visualisation using charts

🢣 Exploratory Data Analysis : 🢱
```
sns.heatmap(df.isnull(),cbar=True)            
plt.show()
```
### ➤ df.isnull()

This creates a DataFrame of True/False values, the same size as your dataset.

Each cell becomes:

True → if the original value was missing (NaN),False → if thvalue exists

### ➤ sns.heatmap(df.isnull(), cbar=True)

sns.heatmap() visualizes that True/False table as a colored grid:

Cells with True (missing) might appear in bright color (like yellow or white).

Cells with False (not missing) appear dark (like black or blue).

cbar=True shows a color bar on the side that indicates what the colors represent (True/False).

---

```

df.hist(figsize=(10,8))
plt.show()

```
### ➤ df.hist()

This function automatically creates histograms for all numeric columns in your dataset.

A histogram shows how values are distributed:

The X-axis → the value range (like age 10–50).

The Y-axis → how many times each range appears.

You can quickly see:

Which ranges are most common

Whether the data is skewed (leaning to one side)

If there are outliers

### ➤ figsize=(10,8)

This sets the width and height of the plot.

---
```
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')

plt.show()
```
### ➤ sns.heatmap( annot=True, cmap='coolwarm')

Displays this correlation matrix as a colored grid:

Each square shows the correlation between two numeric features.

annot=True prints the correlation numbers inside each cell.

cmap='coolwarm' uses a red–blue color gradient:

Blue = positive correlation

Red = negative correlation

---
🢧 Data wrangling
