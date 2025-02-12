from utils.preprocess import *
import multiprocessing

RESULTS_PATH = "../../massivekb_dataset/results"
if __name__ == '__main__':
    chronologer = load_dataframe("datasets/chronologer.tsv")
    #calibrate_directory(RESULTS_PATH+'/first_100_extended/psms', RESULTS_PATH+'/first_100_extended_calibrated', chronologer)
    calibrate_directory('test_data','test_data_cal',
                        chronologer)
