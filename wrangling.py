#data wrangling in pyhton
import pandas as pd
data = {
    'IDNO':[100,102,103,104],
    'Name':['aura','prince','joshua','aurora'],
    'Age':[20,23,21,22],
    'Gender':['Female','Male','Male','Female'],
    'Marks':[76,74,'NaN',78]
}
aura = pd.DataFrame(data)
print(aura)
c=avg=0
for me in aura['Marks']:
    if str (me).isnumeric():
        c += 1
        avg += me
        avg /= c
        aura = aura.replace(to_replace='NaN',value=avg)
        print("\n== New_Data ==\n")
        print(aura)
        
#Filtering out some data :

#aura=aura[aura['Marks']>=75]
#aura= aura.drop(['Age'],axis=1)
#print("\n=== Filtered_Data ===\n")
#print(aura)

#suppose we want to prin tthe top 2 students

aura.sort_values(by='Marks',ascending=False,inplace=True)
top_scorers = aura[['Name','Age','Gender','Marks']].head(2)
print(top_scorers)

#using merge operation to merge two data sets: 
FEEDETAILS = {

    'IDNO':[100,102,103,104],
    'PENDING':['400','NULL','300','NULL']
 }
aurab =pd.DataFrame(FEEDETAILS)
print(aurab)
#merging:
print(pd.merge(aura,aurab,on='IDNO'))#since IDNO is common