import numpy as np
from src import lzc
import mne
import os


# This file will load all files in the data directory ending with the .edf
# extension and save the final values in a vector called lzc_all_edf.

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

# Saving variable to a .csv in the results/ directory

saveDir = 'results/'
np.savetxt(saveDir + "lzc_all_edf.csv", lz_all_edf, delimiter="," )















