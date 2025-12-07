import pandas as pd
import mysql.connector

connection = mysql.connector.connect(
    host = 'localhost'
    user = 'root'
    password = 'admin123'
    database = 'fraud_analysis'
)
query = "SELECT * FROM transactions";
aura=pd.read_sql(query, connection)
print("DataFrame shape :",df.shape)
print(df.head(10))
connection.close()