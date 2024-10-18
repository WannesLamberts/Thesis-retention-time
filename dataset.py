
import requests  # Used for making HTTP requests to retrieve data from web pages or APIs
from bs4 import BeautifulSoup  # Used for parsing HTML and XML documents, especially for web scraping
import os  # Provides a way of interacting with the operating system, such as file and directory manipulation
import zipfile  # Used for working with zip archives, including creating, extracting, and reading zip files
import pandas as pd  # Provides powerful data structures and data analysis tools, especially for working with data in tabular format (DataFrames)
import re  # Provides regular expression matching operations for working with strings
import shutil  # Used for high-level file operations, such as copying, moving, and removing files and directories
import numpy as np  # Provides support for large, multi-dimensional arrays and matrices, along with mathematical functions to operate on these arrays
from sklearn import linear_model  # Part of the scikit-learn library, used for implementing linear regression models and other machine learning algorithms
import matplotlib.pyplot as plt  # A plotting library used for creating static, animated, and interactive visualizations in Python

#link towards the dataset
LINK_DATASET = "https://ftp.pride.ebi.ac.uk/pride/data/archive/2017/02/PXD004732/"

def scrape_dataset(url,directory_path,max=None):
    """
    Scrapes and downloads the zip files on the website located at url.

    Parameters:
    -----------
    url : str
        The URL of the webpage where the dataset links are located. The webpage should contain downloadable `.zip` files.

    directory_path : str
        The path to the directory where the downloaded files will be saved. The directory will be created if it does not exist.

    max : int, optional
        The maximum number of files to download. If `None`, all available `.zip` files will be downloaded.
    """

    # Send a GET request to retrieve the webpage content
    response = requests.get(url)

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract all links ending with '.zip'
    links = [url + '/' + node.get('href') for node in soup.find_all('a')
             if node.get('href').endswith('.zip')]

    # Limit the number of links if max is specified
    if max:
        links = links[:max]

    # Create the target directory if it does not exist
    os.makedirs(directory_path, exist_ok=True)

    # Loop through the links and download each file
    for link in links:
        filename = os.path.join(directory_path, link.split('/')[-1])
        with requests.get(link, stream=True) as r:
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Downloaded {filename}")
    return

def extract_file_from_zip(directory_path,zip_path,target_file_name="evidence.txt"):
    """
    Extracts a specified file from ZIP archives and saves it to a designated directory.

    Parameters:
    -----------
    directory_path : str
        The path to the directory where the extracted files will be saved. If the directory does not exist, it will be created.

    zip_path : str
        The path to the directory containing the ZIP files. The function will search for ZIP files in this directory.

    target_file_name : str, optional
        The name of the file to be extracted from the ZIP archives. By default, it is set to 'evidence.txt'.
        Only files that match this name will be extracted.
    """

    # Create a directory to save the extracted files
    os.makedirs(directory_path, exist_ok=True)

    # Loop through each ZIP file in the 'data' folder
    for zip_filename in os.listdir(zip_path):
        if zip_filename.endswith('.zip'):
            zip_filepath = os.path.join(zip_path, zip_filename)

            # Open the ZIP file
            with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
                # Loop through all files in the ZIP archive
                for file in zip_ref.namelist():
                    # Check if the file matches the target file name
                    if file == target_file_name:
                        # Extract the file to the specified directory
                        extracted_path = os.path.join(directory_path,zip_filename[:-4]+"-"+target_file_name)
                        with open(extracted_path, 'wb') as f_out:
                            f_out.write(zip_ref.read(file))
                        print(f"Extracted {file} to {extracted_path}")

    print("All 'evidence' files have been extracted.")
    return


def merge_tabular_files(directory_path, output_file,seperator='\t'):
    """
    Merges all tabular files in a specified directory into a single file and saves the result.

    Parameters:
    -----------
    directory_path : str
        The path to the directory containing the tabular files that will be concatenated.

    output_file : str
        The path where the merged content will be saved. The result will be written to this file.

    separator : str, optional
        The delimiter used to separate columns in the tabular files. By default, it is set to '\t' (tab-separated).
        You can change this to other delimiters like ',' for comma-separated files.
    """
    # List to store the dataframes from each file
    all_data = []

    # Loop through each file in the directory
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        print(f"Processing {file_path}...")

        # Read the file into a pandas DataFrame
        df = pd.read_csv(file_path, sep=seperator)  # Assuming tab-separated values
        all_data.append(df)

    # Concatenate all the dataframes into one
    combined_df = pd.concat(all_data, ignore_index=True)

    # Save the combined data to the output file
    combined_df.to_csv(output_file, sep='\t', index=False)

    print(f"All files have been concatenated into {output_file}.")


def load_dataframe(path_csv,columns=None,seperator='\t'):
    """
    Loads a CSV file into a pandas DataFrame, with an option to keep only selected columns.

    Parameters:
    -----------
    path_csv : str
        The path to the CSV file that needs to be loaded.

    columns : list, optional
        A list of column names to retain in the DataFrame. If not provided, all columns will be loaded.
        This allows you to select specific columns of interest from the dataset.

    separator : str, optional
        The delimiter used to separate values in the CSV file. By default, it is set to '\t' (tab-separated values).
        You can specify other delimiters such as ',' for comma-separated files.

    Returns:
    --------
    pandas.DataFrame
        A pandas DataFrame containing the loaded data, with the specified columns if provided.
    """
    df = pd.read_csv(path_csv, sep=seperator)
    if columns:
        df = df[columns]
    return df

def preprocess_dataframe(df,format_modified_sequence = True,min_score = 90, max_PEP = 0.01,reset_index = True):
    """
    Preprocess the input DataFrame by formatting the 'Modified sequence' column,
    and filtering based on the minimum score and maximum PEP values. At the end resets the index if chosen.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to be preprocessed.

    format_modified_sequence : bool, optional
        If True, removes the first and last character of the 'Modified sequence' column. Default is True.

    min_score : float, optional
        The minimum score threshold. Rows with a score less than this value will be removed. Default is 90.

    max_PEP : float, optional
        The maximum PEP threshold. Rows with PEP greater than this value will be removed. Default is 0.01.

    reset_index : bool, optional
        If True, resets the index of the dataframe, Default is True.

    Returns:
    --------
    pandas.DataFrame
        The preprocessed DataFrame
    """

    # format the 'Modified sequence' by removing the first and last character
    if format_modified_sequence:
        df["Modified sequence"] = df["Modified sequence"].str[1:-1]

    # Filter rows based on the 'Score' and 'PEP' columns
    df = df[df["Score"]>=min_score]
    df = df[df["PEP"]<=max_PEP]

    # Reset the index and drop the old index
    if reset_index:
        df = df.reset_index(drop=True)

    return df


def preprocess_directory(target_dir_path,output_dir_path,columns = ["Modified sequence","Retention time","Score","PEP","Experiment"],format_modified_sequence = True,min_score = 90, max_PEP = 0.01,reset_index = True):
    """
    Preprocesses all CSV files in the specified directory by filtering and formatting their contents.
    The processed files are written to the specified output directory.

    Parameters:
    -----------
    target_dir_path : str
        The path to the directory containing the CSV files to be processed.

    output_dir_path : str
        The path to the directory where the processed files will be saved. If the directory
        does not exist, it will be created.

    columns : list, optional
        A list of column names to retain in each file. The default columns are:
        "Modified sequence", "Retention time", "Score", "PEP", and "Experiment".

    format_modified_sequence : bool, optional
        If True, applies a specific formatting function to the "Modified sequence" column.
        Default is True.

    min_score : float, optional
        The minimum score threshold for filtering rows. Only rows with a score greater than or
        equal to this value will be retained. Default is 90.

    max_PEP : float, optional
        The maximum Posterior Error Probability (PEP) threshold for filtering rows. Only rows with
        a PEP less than or equal to this value will be retained. Default is 0.01.

    reset_index : bool, optional
        If True, resets the index of the DataFrame after filtering. Default is True.

    """
    os.makedirs(output_dir_path, exist_ok=True)
    for filename in os.listdir(target_dir_path):
        path = os.path.join(target_dir_path,filename)
        df = load_dataframe(path,columns)
        df = preprocess_dataframe(df,format_modified_sequence,min_score,max_PEP,reset_index)
        output_path = os.path.join(output_dir_path,filename)
        write_dataframe_to_file(df,output_path)

def write_dataframe_to_file(df, output_path, separator='\t'):
    """
    Writes a pandas DataFrame to a CSV file with an optional custom separator.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame that needs to be written to the CSV file.

    output_path : str
        The file path where the CSV file will be saved.

    separator : str, optional
        The delimiter used to separate values in the CSV file. By default, it is set to '\t' (tab-separated values).
        You can specify other delimiters such as ',' for comma-separated files.
    """
    try:
        df.to_csv(output_path, index=False, sep=separator)
        print(f"DataFrame successfully written to {output_path} with separator '{separator}'")
    except Exception as e:
        print(f"An error occurred while writing the DataFrame to CSV: {e}")

def sort_evidence_files(location):
    """
    Sorts evidence files into directories based on a pool number extracted from their filenames.

    Parameters:
    -----------
    location : str
        The path to the directory where the evidence files are stored.
        All files in this directory will be checked, and sorted into subdirectories.
    """
    for filename in os.listdir(location):
        # Search for "Pool_<number>" in the filename (case-insensitive)
        pool = re.search("Pool_(\\d+)", filename,re.IGNORECASE).group(0).capitalize()

        # Define the path of the pool directory and file location
        directory = os.path.join(location, pool)
        file_location = os.path.join(location,filename)
        os.makedirs(directory, exist_ok=True)

        #move file to the directory
        shutil.move(file_location,directory)
    print("Evidence files sorted")


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
        overlap = df.merge(calibration_df, how="inner", on="Modified sequence")
        return {
            k: v for k, v in zip(overlap["Modified sequence"], overlap["iRT"])
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



#scrape_dataset(LINK_DATASET,"zips")
#extract_file_from_zip("data","zips")
#merge_tabular_files("data", "evidence_combined.tsv")
#df = load_dataframe("evidence_combined.tsv",["Modified sequence","Retention time","Score","Experiment"])
#write_dataframe_to_file(df,"simple_dataframe.tsv")
#sort_evidence_files("data")
#df = load_dataframe("testing/data_small/Pool_2/Thermo_SRM_Pool_2_01_01_2xIT_2xHCD-1h-R2-tryptic-evidence.txt",["Modified sequence","Retention time","Score","PEP","Experiment"])
#print(df["Retention time"].head())
#df = preprocess_dataframe(df)
#df= calibrate_to_iRT(df,plot = False)
#print(df[["Retention time","iRT"]].head())
#preprocess_directory(r"D:\data_Original", "processed")
