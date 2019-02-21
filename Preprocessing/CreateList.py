import numpy as np
import sys
import os
from scipy.io import wavfile
# from python_speech_features import mfcc

# Read args
# Label_text_source = sys.argv[1];
# Output_dir = sys.argv[2];

# f = open(Output_dir + 'train.lst','w')

# for filename in os.listdir(Label_text_source):
#   filename_parts = filename.split('.')
#   if (filename_parts[1] == "wav"):
#       f.write(filename + '\n')

# f.close() 

def getDataSource(data_root_dir, list_dir):
    with open(list_dir, 'w') as wdir:
        for root, subdir, files in os.walk(data_root_dir):
            for f in files:
                fname, fext = os.path.splitext(f)
                if (fext == '.wav'):
                    wdir.write(os.path.join(root, f) + '\n')


if __name__ == '__main__':
    data_dir = sys.argv[1]
    list_dir = sys.argv[2]
    getDataSource(data_dir, list_dir)
