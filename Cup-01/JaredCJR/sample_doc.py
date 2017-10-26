#!/usr/bin/python3
import numpy as np
import pandas as pd

cut_programs = np.load('../data/cut_Programs.npy')
voc_dict = np.load('./voc_dict.npy')

sample_doc = []
for cut_program in cut_programs:
    for episode in cut_program:
        for line in episode:
            sample_line = ''
            for w in line:
                if w in voc_dict:
                    sample_line += w+' '
            sample_doc.append(sample_line)

np.save('./sample_doc.npy', sample_doc)
print(sample_doc[:10])

