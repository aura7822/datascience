#==============================
#|                            |
#| EXPLARATORY DATA ANALYSIS  |
#|                            |
#==============================
#CODE FOR EPLARATORY DATA ANALYSIS

import pandas as pd #data manipulation and statistical summary
import seaborn as sns # advanced aesthetic plots
import matplotlib.pyplot as plt # basic plotting

aura = pd.read_csv('/home/aura/Documents/2.2/datascience/data.csv') #reads command separated values from path in pc
print(aura.shape) # Responsible for dimensions ,rows and colums
print(aura.info()) #prints data types and checks for missing values
print(aura.describe()) #shows  satistical summaries sof numerical colums

sns.heatmap(aura.isnull(),cbar = True)
plt.show()

aura.hist(figsize=(10,0))
plt.show()
sns.heatmap(aura.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.show()
