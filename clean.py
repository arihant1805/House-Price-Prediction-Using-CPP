import pandas as pd



data = pd.read_csv('data.csv')
data = data.iloc[:,1:10]
data = data.dropna()
data = data.drop_duplicates()

data.to_csv('data_cleaned.csv', index=False)