from utils.preprocess import *
import multiprocessing

RESULTS_PATH = "../../massivekb_dataset/working_results"
if __name__ == '__main__':
    chronologer = load_dataframe("datasets/chronologer.tsv")
    calibrate_directory(RESULTS_PATH+'/results/psms', RESULTS_PATH+'/results_calibrated2', chronologer)
