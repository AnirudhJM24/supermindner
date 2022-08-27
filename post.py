import pandas as pd

df = pd.read_csv('extracted.csv')
df = df[df['keywords']!='[]']
df.to_csv('features_extracted.csv',index = False)