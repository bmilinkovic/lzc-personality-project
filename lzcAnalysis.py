from src import lzc
import mne
import os


# Get the data for 1 participant. Here ideally, you would get the data of all participants, store them somehow
# for later analysis pipeline
dataDir = "data/"
data = mne.io.read_raw_bdf(os.path.join(dataDir, "P1.bdf"))

# To extract JUST the raw data file as a numpy array, or 'matrix' we need to select this from the raw data structure
# that MNE creates, so lets get that:

raw_array = data[:][0]

# Now begin Lempel-Ziv Analysis:
# As above, here you would ideally create a for loop that will loop over every subject.
# I actually think that all of it can be put in a for loop,
# i.e.
#   for each file in the data folder:
#           load that file
#           extract the raw numpy array 'matrix' for Lzc computation
#           compute LZc on that multidimensional array
#           save the lempelziv complexity into a variable and keep appending this variable with new quantities.
#           so then we have a vector valued variable of all lempel-ziv complexities for each participant.
#   end for loop.

lzc.LZc(raw_array)

