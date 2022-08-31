# importing all the modules we need

from src import lzc
import mne
import os
from pathlib import Path

# Loop over all participants in data/ directory in project.
# The loop also calculates the total LZc for each participant's multidimensional time-series and saves in the variable
# lz_all
dataDir = "data/"
ext = ".bdf"
lz_all = []
for filename in os.listdir(dataDir):
    if filename.endswith(ext):
        f = os.path.join(dataDir, filename)
        data = mne.io.read_raw_bdf(f)
        raw_array = data[:][0]
        LZc_single_subject = lzc.LZc(raw_array)
        lz_all.append(LZc_single_subject)





