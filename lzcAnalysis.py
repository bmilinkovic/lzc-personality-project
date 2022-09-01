# importing all the modules we need

import numpy as np
from src import lzc
import mne
import os
import pandas as pd

# Loop over all participants in data/ directory in project.
# The loop also calculates the total LZc for each participant's multidimensional time-series and saves in the variable
# lz_all_edf for edf files and lz_all_csv for csv files.

# 1. Using .edf files
dataDir = "data/"
ext_edf = ".edf"
lz_all_edf = []

for filename in os.listdir(dataDir):
    if filename.endswith(ext_edf):
        f = os.path.join(dataDir, filename)
        data = mne.io.read_raw_edf(f)
        raw_array = data[:][0]
        LZc_single_subject_edf = lzc.LZc(raw_array)
        lz_all_edf.append(LZc_single_subject_edf)


# 2. using transposed csv files.
ext_csv = ".csv"
lz_all_csv = []
for filename in os.listdir(dataDir):
    if filename.endswith(ext_csv):
        f = os.path.join(dataDir, filename)
        file_csv = pd.read_csv(f).to_numpy().T
        LZc_single_subject_csv = lzc.LZc(raw_array)












