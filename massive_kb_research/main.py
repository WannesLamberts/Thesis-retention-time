from utils.preprocess import *
import multiprocessing

RESULTS_PATH = "../../massivekb_dataset/results"
if __name__ == '__main__':
    chronologer = load_dataframe("datasets/chronologer.tsv")
    calibrate_directory(RESULTS_PATH+'/test_data/psms', RESULTS_PATH+'/test_data_calibrated', chronologer)
