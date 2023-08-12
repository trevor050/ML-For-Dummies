#this simple script will remove all NaN values in a csv file and output a new csv file
import pandas as pd

df = pd.read_csv('IMDB.csv', delimiter=',', encoding='utf-8', error_bad_lines=False)
df = df.dropna()
df.to_csv('IMDBCleaned.csv', index=False)