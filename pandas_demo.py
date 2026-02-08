import numpy as np
import pandas as pd

data = {
        'Name':['Alice','Bob','Charlie'],
        'Age':[25,30,55],
        'Score':[90,78,99]
        }
df_raw = pd.DataFrame(data)
print(df_raw)

df = pd.read_csv(r"C:\Users\thack\OneDrive\Documents\Python\students.csv")
# See first few rows
#print(df.head())
# See last few rows
#print(df.tail())
# See random sample
#print(df.sample(5))
print(df.describe())
