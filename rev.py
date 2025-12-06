# basic pandas operations : 

import pandas as pd

data = {
    'Name': ['Prince','Joshua','Aura','Julia'],
    'Age': [20,21,26,28],
    'Salary':[10000,11000,18000,22000],
    'Department':['HR','ICT','ENGINEERING','QUALITYASSURANCE']
}
aura= pd.DataFrame(data)
print(aura)
print("\n")

#Filtering by selecting rows and dept :

filtered_aura = aura[(aura['Age']>25) & (aura['Department']=='ICT')]
print("Filtered version:")

#soting actions:
sorted_aura = aura.sort_values(by='salary', ascending=False)
print("Sorted salary : ")