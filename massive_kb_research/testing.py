from utils.preprocess import *


merge_psm_files("test_data/first_two", "output/out.tsv")

df = load_dataframe('output/out.tsv')
grouped = df.groupby('filename')
chronologer = load_dataframe("datasets/chronologer.tsv")


calibration_peptides_base = get_calibration_peptides(df)
calibration_peptides_chronologer = get_calibration_peptides(df,chronologer)

# print(len(calibration_peptides_chronologer))
# if len(calibration_peptides_chronologer) == 0:
#     print("NO OVERLAP")
# else:
#     for filename, group in grouped:
#         overlap = check_overlap(group, calibration_peptides_chronologer.keys())
#         print(f"filename {filename} has {overlap} calibration peptides" )


df2 = grouped.apply(lambda group: calibrate_to_iRT(group, chronologer, 'PeptideModSeq','RT'),include_groups=False).reset_index()
df2 = df2.drop(columns='level_1')
write_dataframe_to_file(df2,"output/calibrated.tsv")
print(len(df))
print(len(df2))
print(df.columns)
print(df2.columns)

# df_test = df[df["filename"]=="20111219_EXQ5_KiSh_SA_LabelFree_HeLa_Proteome_Control_rep2_pH5.mzML"]
# df_out = calibrate_to_iRT(df_test,chronologer, 'PeptideModSeq','RT')
# write_dataframe_to_file(df_out,"output/calibratedV3.tsv")
