import sys

from utils.preprocess import *
import multiprocessing

if __name__ == '__main__':
    psms_path = sys.argv[1]
    output_path = sys.argv[2]
    chronologer = load_dataframe("datasets/chronologer.tsv")
    calibrate_directory(psms_path, output_path, chronologer)
