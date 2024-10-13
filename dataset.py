
import requests
from bs4 import BeautifulSoup
import os
import zipfile
import pandas as pd

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

#scrape_dataset(LINK_DATASET,"zips",1)
#extract_file_from_zip("data","zips")
#merge_tabular_files("data", "evidence_combined.csv")
df = load_dataframe("evidence_combined.csv",["Modified sequence","Retention time","Score","Experiment"])
print(df.head())