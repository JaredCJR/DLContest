import numpy as np
# import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer


class WordPreprocess(object):
    
    def __init__(self):
        pass

# ===============================================================================
# question_parse:
# Input: 
# qeusitons - required, input data from cut_program.npy
# program - optional, input data from ex_program.npy
# word_dict- required, input data from voc_dict.npy, which is a word dictionary
# 
# ===============================================================================
    def question_parse(self, questions=None, program=None, word_dict=None):
        if (questions is None or word_dict is None):
            print ("NOT ENOUGH INPUT, ABORT!")
            return

        temp_prog = np.ndarray(shape=(1, 0))

        for question in questions:
            for i, line in enumerate(question):
                if (i == 0):
                    ex_line = ''
                    for qline in question[i]:
                        for l in qline:
                            for w in l:
                                if (w in word_dict):
                                    ex_line += w+' '
                    if (program is not None):
                        program = np.append(program, ex_line)
                    else:
                        temp_prog = np.append(temp_prog, ex_line)

                else:
                    ex_line = ''
                    for w in line:
                        if w in word_dict:
                            ex_line += w + ' '
                    if (program is not None):
                        program = np.append(program, ex_line)
                    else:
                        temp_prog = np.append(temp_prog, ex_line)

        return (program if program is not None else temp_prog)

# ===============================================================================
# TF-IDF transformer
# Parameter:
# input_ - input program which is going to transfrom
# min_gram - min gram number
# max_gram - max gram number
# ===============================================================================

    def tfidf_transform(self, input_, max_gram=1, min_gram=1):
        self.tfidf = TfidfVectorizer(ngram_range=(min_gram, max_gram))
        self.tfidf.fit(input_)
        doc_tfidf = self.tfidf.transform(input_).toarray()

        return doc_tfidf

# ===============================================================================
# Feature hash
# Parameter:
# input_ - input program which is going to transform
# num_feauture - output feature number
# ===============================================================================

    def hash_transform(self, input_, num_feature=128):
        self.f_hash = HashingVectorizer(n_features=num_feature)
        doc_hash = self.f_hash.transform(input_).toarray()

        return doc_hash

