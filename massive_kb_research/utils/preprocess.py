import multiprocessing
import os


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import linear_model


def load_dataframe(path, columns=None,has_header=True,separator='\t'):
    """
    Loads a file into a pandas DataFrame with an option to keep only selected columns,
    and allows you to choose if the file contains headers.

    Parameters:
    -----------
    path : str
        The path to the file that needs to be loaded.

    columns : list, optional
        A list of column names to set if the dataframe doesn't have a header.
        Only used when `has_header=False`.

    separator : str, optional
        The delimiter used to separate values in the file. By default, it is set to '\t' (tab-separated values).
        You can specify other delimiters such as ',' for comma-separated files.

    has_header : bool, optional
        Whether the file contains header information. Default is `True`.

    Returns:
    --------
    pandas.DataFrame
        A pandas DataFrame containing the loaded data, with the specified columns if provided.
    """

    if has_header:
        df = pd.read_csv(path, sep=separator)
    else:
        if columns is None:
            raise ValueError("If `has_header` is False, you must provide column names using the 'columns' parameter.")
        # Read the file without headers, use the provided column names
        df = pd.read_csv(path, sep=separator, header=None, names=columns)

    return df

def merge_psm_files(directory, out_file,header = ['filename','scan','RT','sequence','mztab_filename','task_id']):
    """
    Merges all tsv files containing psms in the given directory in one tsv file

    Parameters:
    -----------
    directory : str
        The path to the directory containing the tsv files to be merged.

    out_file : str
        The filename of the merged tsv file.

    header : list
        A list of column names to be used as the header in the merged TSV file.
    """
    files = sorted(
        [f for f in os.listdir(directory) if f.endswith(".tsv") and os.path.isfile(os.path.join(directory, f))])

    with open(out_file, "w", encoding="utf-8") as outfile:
        outfile.write("\t".join(header) + "\n")
        for file in files:
            file_path = os.path.join(directory, file)
            with open(file_path, "r", encoding="utf-8") as infile:
                outfile.write(infile.read())

    print(f"All TSV files have been merged into {out_file}")



def check_overlap(df, cal_peptides):
    """
    checks the amount of overlap there is from df with dict

    """
    group_values = set(df["PeptideModSeq"])
    overlap = group_values.intersection(cal_peptides)
    return len(overlap)



def get_calibration_peptides(df, calibration_df=None):
    """
    Retrieves a dictionary of calibration peptides and their corresponding iRT (indexed Retention Time) values.

    Author:
    -----------
    Ceder Dens

    Parameters:
    -----------
    df : pandas.DataFrame
        The main DataFrame containing peptide data.

    calibration_df : pandas.DataFrame, optional
        A DataFrame containing reference peptides and their known iRT values. If provided, the function
        will return calibration peptides that overlap between `df` and `calibration_df`. If not provided,
        a default set of iRT calibration peptides will be used.

    Returns:
    --------
    dict
        A dictionary where the keys are peptide sequences (str) and the values are the corresponding iRT values (float).
        If `calibration_df` is provided, the dictionary will contain peptides from the overlap of `df` and `calibration_df`.
        Otherwise, a predefined set of calibration peptides and iRT values is returned.
    """
    if calibration_df is None:
        return {
            "TFAHTESHISK": -15.01839514765834,
            "ISLGEHEGGGK": 0.0,
            "LSSGYDGTSYK": 12.06522819926421,
            "LYSYYSSTESK": 31.058963905737304,
            "GFLDYESTGAK": 63.66113155016407,
            "HDTVFGSYLYK": 72.10102416227504,
            "ASDLLSGYYIK": 90.51605846673961,
            "GFVIDDGLITK": 100.0,
            "GASDFLSFAVK": 112.37148254946804,
        }
    else:
        overlap = df.merge(calibration_df, how="inner", on="PeptideModSeq")
        return {
            k: v for k, v in zip(overlap["PeptideModSeq"], overlap["Prosit_RT"])
        }

def calibrate_to_iRT(df,calibration_df=None,seq_col="Modified sequence",rt_col="Retention time",
    irt_col="iRT",plot=False,filename=None,take_median=False,):
    """
    Calibrates the retention times in a DataFrame to indexed Retention Time (iRT) values using a set of calibration peptides.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing peptide sequences and their respective retention times

    calibration_df : pandas.DataFrame, optional
        A DataFrame containing calibration peptides and their known iRT values. If not provided, a predefined
        set of calibration peptides will be used.

    seq_col : str, optional
        The column name in `df` that contains the peptide sequences. Default is "Modified sequence".

    rt_col : str, optional
        The column name in `df` that contains the retention time values. Default is "Retention time".

    irt_col : str, optional
        The column name where the predicted iRT values will be stored in `df`. Default is "iRT".

    plot : bool, optional
        If True, a scatter plot of the original Retention time values vs. iRT values will be generated along with the fitted regression line.

    filename : str, optional
        If provided, the function will print the number of calibration peptides found in the DataFrame. Useful for logging or debugging.

    take_median : bool, optional
        If True, the median retention time for each calibration peptide will be used. Otherwise, the first occurrence of the Retention time value will be used.

    Returns:
    --------
    pandas.DataFrame or None
        The input DataFrame with an additional column containing the calibrated iRT values.
        If fewer than two calibration peptides are found in the input data, the function returns `None`.

    Process:
    --------
    1. The function first retrieves a dictionary of calibration peptides and their corresponding iRT values.
    2. It loops through the calibration peptides and retrieves the corresponding Retention time values from the input DataFrame.
    3. If `take_median` is True, it uses the median Retention time value for each peptide; otherwise, it uses the first occurrence.
    4. The old Retention time values and iRT values are then used to fit a linear regression model.
    5. The model is used to predict iRT values for all peptides in the input DataFrame.
    6. If `plot` is True, a scatter plot of calibration points and the regression line is displayed.
    7. The function returns the input DataFrame with an additional column for iRT values, or `None` if calibration fails.
    """

    # Get calibration peptides and their corresponding iRT values
    calibration_peptides = get_calibration_peptides(df, calibration_df)
    old_rt = []
    cal_rt = []

    # Loop through each calibration peptide
    for pep, iRT in calibration_peptides.items():
        # Filter the DataFrame to get rows corresponding to the current peptide sequence
        pep_df = df[df[seq_col] == pep]
        if len(pep_df) > 0:
            # Use the median or first occurrence of the RT value based on the `take_median` flag
            if take_median:
                old = np.median(df[df[seq_col] == pep][rt_col])
            else:
                old = df[df[seq_col] == pep][rt_col].iloc[0]

            old_rt.append(old)
            cal_rt.append(iRT)
    # Log the number of calibration peptides found if `filename` is provided
    if filename is not None:
        print(
            f"{filename} had {len(old_rt)}/{len(calibration_peptides)} calibration peptides"
        )
    # If fewer than two calibration peptides are found, return None (unable to perform calibration)
    if len(old_rt) < 2:
        print("NOT ENOUGH CAL PEPTIDES FOUND")
        return None

    # Fit a linear regression model using the original RT values and the iRT values
    regr = linear_model.LinearRegression()
    regr.fit(np.array(old_rt).reshape(-1, 1), np.array(cal_rt).reshape(-1, 1))

    # Predict iRT values for all peptides in the input DataFrame
    df[irt_col] = regr.predict(df[rt_col].values.reshape(-1, 1))

    # Plot the calibration points and the fitted regression line if `plot=True`
    if plot:
        plt.scatter(old_rt, cal_rt, label="calibration points")
        plt.plot(
            range(int(min(old_rt) - 5), int(max(old_rt) + 5)),
            regr.predict(
                np.array(
                    range(int(min(old_rt) - 5), int(max(old_rt) + 5))
                ).reshape(-1, 1)
            ),
            label="fitted regressor",
        )
        plt.legend()
        plt.show()

    return df


def process_file(file,directory, out_dir, calibration_df):
    """
    calibrates the file in the directory.
    the file holds a dataframe which has a column filename.
    The calibrating will be run on each filename.

    Parameters:
    -----------
    file : string
        The name of the tsv file to be calibrated
    directory : string
        The name of the directory containing the tsv files.

    out_dir : string
        The name of the directory where the calibration results will be saved.

    calibration_df : pandas.DataFrame
        The dataframe which will be used as reference for the calibration.

    """
    print(f'started {file}')
    file_path = os.path.join(directory, file)
    df = load_dataframe(file_path,['filename','scan','RT','PeptideModSeq','task_id'],False)

    calibrated_df = df.groupby('filename').apply(
        lambda group: calibrate_to_iRT(group, calibration_df, 'PeptideModSeq', 'RT'),
        include_groups=False
    ).reset_index()
    calibrated_df = calibrated_df.drop('level 1', axis=1)
    output_file_path = os.path.join(out_dir, file)
    calibrated_df.to_csv(output_file_path, sep='\t', index=False,header=False)
    print(f'Calibrated {file}')


def calibrate_directory(directory, out_dir,calibration_df):
    """
    Calibrates the retention time of each tsv file in the directory.


    Parameters:
    -----------
    directory : string
        The name of the directory containing the tsv files.

    out_dir : string
        The name of the directory where the calibration results will be saved.

    calibration_df : pandas.DataFrame
        The dataframe which will be used as reference for the calibration.

    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    files = sorted(
        [f for f in os.listdir(directory) if f.endswith(".tsv") and os.path.isfile(os.path.join(directory, f))])

    # for file in files:
    #     process_file(file, directory, out_dir, calibration_df)
    num_processes =multiprocessing.cpu_count()-1
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.starmap(process_file, [(file, directory, out_dir, calibration_df) for file in files])
    ##



def write_dataframe_to_file(df, output_path, separator='\t'):
    """
    Writes a pandas DataFrame to a file with an optional custom separator.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame that needs to be written to the file.

    output_path : str
        The file path where the file will be saved.

    separator : str, optional
        The delimiter used to separate values in the file. By default, it is set to '\t' (tab-separated values).
        You can specify other delimiters such as ',' for comma-separated files.
    """
    try:
        df.to_csv(output_path, index=False, sep=separator)
        #print(f"DataFrame successfully written to {output_path} with separator '{separator}'")
    except Exception as e:
        pass
        #print(f"An error occurred while writing the DataFrame to file: {output_path}")

