import pyarrow.parquet as pq
import os as os
import subprocess
import pandas as pd
class Dataset_Manager():
    def split_in_subsets(self,file_path,output_path,chunk_size = 10000,max_chunks=10):
        table = pq.read_table(file_path)
        print(len(table))
        return
        for i, start_row in enumerate(range(0, table.num_rows, chunk_size)):
            if i >= max_chunks:
                break
            chunk = table.slice(start_row, chunk_size)
            pq.write_table(chunk, f"{output_path}/subset_{i}.parquet")
            print(f"Processed chunk {i}, starting row: {start_row}")
    def download_massive_file(self,massive_filename: str, dir_name: str) -> None:
        """
        Download the given file from MassIVE.

        The file is downloaded using a `wget` subprocess.
        If the file already exists it will _not_ be downloaded again.

        Parameters
        ----------
        massive_filename : str
            The local MassIVE file link.
        dir_name : str
            The local directory where the file will be stored.
        """
        peak_filename = os.path.join(dir_name, massive_filename.rsplit('/', 1)[-1])
        if not os.path.isfile(peak_filename):
            if not os.path.isdir(dir_name):
                try:
                    os.makedirs(dir_name)
                except OSError:
                    pass
            #logger.debug('Download file %s', massive_filename)
            url = f'ftp://massive.ucsd.edu/{massive_filename}'
            proc = subprocess.run(
                ['wget', '--no-verbose', '--timestamping', '--retry-connrefused',
                 f'--directory-prefix={dir_name}', '--passive-ftp', url],
                capture_output=True, text=True)
    def download_massivekb_peaks(self,massivekb_filename: str, dir_name: str) -> None:
        """
        Download all peak files listed in the given MassIVE-KB metadata file.

        Peak files will be stored in subdirectories the given directory per
        dataset.
        Existing peak files will _not_ be downloaded again.

        Parameters
        ----------
        massivekb_filename : str
            The metadata file name.
        dir_name : str
            The local directory where the peak files will be stored.
        """
        filenames = (pd.read_csv(massivekb_filename, sep='\t',
                                 usecols=['filename'])
                     .drop_duplicates('filename'))
        datasets = filenames['filename'].str.split('/', n = 1).str[0]
        filenames['dir_name'] = datasets.apply(
            lambda dataset: os.path.join(dir_name, dataset))
        for filename, dir_name in zip(filenames['filename'],filenames['dir_name']):
            self.download_massive_file(filename, dir_name)

