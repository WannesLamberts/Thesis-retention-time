from utils.preprocess import *
df = pd.read_csv('te.tsv', sep='\t', index_col=False)
print(len(df))
print(df.columns)
print(df['sequence'].nunique())