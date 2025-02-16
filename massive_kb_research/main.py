import sys

from utils.preprocess import *
import multiprocessing

if __name__ == '__main__':
    psms_path = sys.argv[1]
    output_path = sys.argv[2]
    chronologer_loc = sys.argv[3]
    chronologer = load_dataframe(chronologer_loc)
    calibrate_directory(psms_path, output_path, chronologer)
