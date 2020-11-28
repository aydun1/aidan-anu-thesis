import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from scipy.stats import spearmanr, percentileofscore

from keras import backend as K
from keras.models import Model
from keras.layers import Input, LeakyReLU, Concatenate
from keras.layers.merge import Multiply
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, AveragePooling1D, Convolution2D, AveragePooling2D, MaxPooling2D
from keras.optimizers import RMSprop
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

def calc_r2(x, y):
    coeffs = np.polyfit(x, y, deg=1)
    results = {'polynomial': coeffs.tolist()}
    p = np.poly1d(coeffs)
    yhat = p(x)
    ybar = np.sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)
    sstot = np.sum((y - ybar)**2)
    results['determination'] = ssreg / sstot
    return results

def preprocess(df):
    seq_n = 34
    SEQ = np.zeros((seq_n, 4), dtype=int)
    seq = df[seq_col].upper()
    for p in range(seq_n):
        SEQ[p, 'ACGT'.index(seq[p])] = 1
    return SEQ, df[chromatin_col] * 100

def preprocess_seq(seq):
    seq = seq.upper()
    SEQ = np.zeros((len(seq), 4), dtype=int)
    for i, p in enumerate(seq):
        SEQ[i, 'ACGT'.index(p)] = 1
        #SEQ[i]['ACGT'.index(p)] = 1
    return SEQ

## Preprocessor from deepcpf1
def PREPROCESS(lines):
    data_n = len(lines)
    SEQ = np.zeros((data_n, 34, 4), dtype=int)
    CA = np.zeros((data_n, 1), dtype=int)
    print(len(CA))
    for l in range(0, data_n):
        data = lines[l].split()
        seq = data[0]
        for i in range(34):
            if seq[i] in "Aa":
                SEQ[l, i, 0] = 1
            elif seq[i] in "Cc":
                SEQ[l, i, 1] = 1
            elif seq[i] in "Gg":
                SEQ[l, i, 2] = 1
            elif seq[i] in "Tt":
                SEQ[l, i, 3] = 1
        CA[l,0] = int(data[1])*100

    return [SEQ, CA]

def reverse_complement(sequence):
    defs = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    return ''.join(reversed([defs[n] if n in defs else n for n in sequence]))

def preprocess_chromatin(ca):
    return ca * 100

label_col = 'Indel frequency at endogenous target site (background subtracted)'
#label_col = 'Indel freqeuncy\n(Background substracted, %)'

seq_col = '34 bp target sequence (4 bp + PAM + 23 bp protospacer + 3 bp)'
chromatin_col = 'Chromatin accessibility (1= DNase I hypersensitive sites, 0 = Dnase I non-sensitive sites)'
f = 'data/41587_2018_BFnbt4061_MOESM39_ESM.xlsx'
col_mappings = {
    'Cell line': 'cell_line',
    'Chromosomal position (hg19)': 'location',
    'Chromosomal Position (hg19)': 'location',
    '34 bp target sequence (4 bp + PAM + 23 bp protospacer + 3 bp)': 'sequence',
    'Indel frequency at endogenous target site (background subtracted)': 'efficiency',
    'Indel frequency at endogenous target site': 'efficiency',
    #'Indel frequency at synthetic target site (background subtracted)': 'efficiency',
    'Chromatin accessibility (1= DNase I hypersensitive sites, 0 = Dnase I non-sensitive sites)': 'chromatin'
}

def get_key(i):
    return col_mappings[i] if i in col_mappings else i