import numpy as np
import pandas as pd
import os
import mne
from src import lzc


# This file will load all files in the data directory ending with the .csv
# extension and save the final values in a vector called lzc_all_csv.
dataDir = "data/"
ext_csv = ".csv"
lz_all_csv = []
for filename in os.listdir(dataDir):
    if filename.endswith(ext_csv):
        f = os.path.join(dataDir, filename)
        file_csv = pd.read_csv(f).to_numpy().T
        LZc_single_subject_csv = lzc.LZc(raw_array)
        lz_all_csv.append(LZc_single_subject_csv)