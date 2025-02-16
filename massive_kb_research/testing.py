from utils.preprocess import *
merge_psm_files('test_data/psms_dups','dups.tsv',['filename','scan','RT','sequence','mztab_filename','task_id'])
df = pd.read_csv('dups.tsv', sep='\t',index_col=False)
print(df.info())
merge_psm_files('test_data/psms_drop_dups','drop_dups.tsv',['filename','scan','RT','sequence','mztab_filename','task_id'])
df = pd.read_csv('drop_dups.tsv', sep='\t',index_col=False)
print(df.info())
