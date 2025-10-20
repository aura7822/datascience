#==============================
#|                            |
#| EXPLARATORY DATA ANALYSIS  |
#|                            |
#==============================
#CODE FOR EPLARATORY DATA ANALYSIS

import panda as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')

print(df.shape)
print(df.info())
print(df.descibe())

sns.heatmap(df.isnull(),char=False)
plt.show()

df.hist(figsize=(10,0))
plt.show()

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()