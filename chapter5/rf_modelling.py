import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import re
import urllib.request
import json
from scipy.stats import spearmanr, pearsonr, percentileofscore
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import confusion_matrix,mean_squared_error
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.utils import compute_class_weight
from collections import Counter

RANDOM_STATE = 2553
nucleotides = ['A', 'C', 'G', 'T']
dinucleotides = [i + j for i in nucleotides for j in nucleotides]
trinucleotides = [i + j + k for i in nucleotides for j in nucleotides for k in nucleotides]
allnucleotides = nucleotides + dinucleotides + trinucleotides


def reverse_complement(sequence):
    defs = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    return ''.join(reversed([defs[n] if n in defs else n for n in sequence]))


def gc_content(seq):
    d = len([s for s in seq if s in 'CcGc']) / len(seq) * 100
    return round(d, 2)


def get_key(i):
    return col_mappings[i] if i in col_mappings else i


def polyfit(x, y):
    coeffs = np.polyfit(x, y, deg=1)
    results = {'polynomial': coeffs.tolist()}
    p = np.poly1d(coeffs)
    yhat = p(x)
    ybar = np.sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)
    sstot = np.sum((y - ybar)**2)
    results['determination'] = ssreg / sstot
    return results


def tokenize_sequence_global(seq, name):
    tokens = dinucleotides
    data = {n: sum(1 for i in range(len(seq)) if seq.startswith(n, i))/(len(seq)-len(n)+1) for n in tokens}
    s = pd.Series(data)
    s = s.reindex(tokens, axis=1)
    s.index = ['{}_{}'.format(name, x) for x in s.index]
    return s


def tokenize_sequence_local(seq, name):
    tokens = nucleotides + dinucleotides + trinucleotides
    data = {f'loc{len(n)}_{i}_{n}': 1  if seq.startswith(n, i) else 0 for i in range(len(seq)) for n in tokens}
    s = pd.Series(data)
    s.index = ['{}_{}'.format(name, x) for x in s.index]
    return s


def process_sequence(df):
    leftside_global_df = df.apply(lambda _: tokenize_sequence_global(_[:4], 'left'))
    rightside_global_df = df.apply(lambda _: tokenize_sequence_global(_[31:], 'right'))
    protospacer_global_df = df.apply(lambda _: tokenize_sequence_global(_[8:31], 'protospacer'))
    protospacer_global_df5 = pd.concat([df.apply(lambda _: tokenize_sequence_global(_[i:i+5], f'protospacer{i}')) for i in range(0,30)], axis=1)
    protospacer_global_df8 = pd.concat([df.apply(lambda _: tokenize_sequence_global(_[i:i+8], f'protospacer{i}')) for i in range(0,27)], axis=1)
    protospacer_global_df = pd.concat([protospacer_global_df, protospacer_global_df5, protospacer_global_df8], axis=1)
    leftside_local_df = df.apply(lambda _: tokenize_sequence_local(_[:4], 'left'))
    rightside_local_df = df.apply(lambda _: tokenize_sequence_local(_[31:], 'right'))
    protospacer_local_df = df.apply(lambda _: tokenize_sequence_local(_[8:31], 'protospacer'))
    protospacer_gc = df.apply(lambda _: gc_content(_[8:31])).rename('gc')
    rolling_gc = pd.concat([df.apply(lambda _: gc_content(_[i:i+7])).rename(f'gc{i}') for i in range(0,28)], axis=1)
    global_features_df = pd.concat([protospacer_global_df, leftside_global_df, rightside_global_df], axis=1)
    local_features_df = pd.concat([protospacer_local_df, leftside_local_df, rightside_local_df], axis=1)
    features_df = pd.concat([global_features_df, local_features_df, rolling_gc, protospacer_gc], axis=1)
    return features_df


## Process mini-human dataset to calculate efficiencies
bfs = pd.read_csv('data/g3/tableS1.txt', delimiter='\t').rename(columns={'Gene': 'gene', 'BayesFactor': 'bf'}).set_index('gene')
mini_meta = pd.read_csv('data/mini/mono_details2.csv').set_index('short_name')
mini_deets = pd.read_csv('data/mono_time_samples.csv').rename(columns={'Unnamed: 0': 'short_name'}).set_index('short_name')
mini_df = mini_meta.join(mini_deets)
mini_df = mini_df.reset_index().set_index('gene').join(bfs).reset_index().set_index('short_name')
new_df = mini_df.reset_index(level=0)
new_df['gene'] = new_df['short_name'].apply(lambda _: _.split('_')[0])
new_df['target'] = new_df['short_name'].apply(lambda _: int(_.split('_')[1]))
new_df.set_index(['gene','target'], inplace=True)
new_df['label'] = (new_df['week2'] / new_df['ref']) * - 1
new_df = new_df.loc[new_df['BF_HCT116']  > 25] # GOOD
new_df = new_df.loc[new_df['BF_HCT116']  < 45] 
new_df = new_df.loc[new_df['week1'] > 2500]
new_df['efficiency'] = ((new_df['label']) * 35 + 70)


## Load efficiencies of endogynoustargets
label_col = 'Indel frequency at endogenous target site (background subtracted)'
seq_col = '34 bp target sequence (4 bp + PAM + 23 bp protospacer + 3 bp)'
chromatin_col = 'Chromatin accessibility (1= DNase I hypersensitive sites, 0 = Dnase I non-sensitive sites)'
f = 'data/41587_2018_BFnbt4061_MOESM39_ESM.xlsx'
col_mappings = {
    'Cell line': 'cell_line',
    'Chromosomal position (hg19)': 'location',
    'Chromosomal Position (hg19)': 'location',
    '34 bp target sequence (4 bp + PAM + 23 bp protospacer + 3 bp)': 'sequence',
    '34 bp synthetic target and target context sequence\n(4 bp + PAM + 23 bp protospacer + 3 bp)': 'sequence',
    'Indel frequency at endogenous target site (background subtracted)': 'efficiency',
    'Indel frequency at endogenous target site': 'efficiency',
    'Indel freqeuncy\n(Background substracted, %)': 'efficiency',
    'Chromatin accessibility (1= DNase I hypersensitive sites, 0 = Dnase I non-sensitive sites)': 'chromatin'
}
lenti_df = pd.read_excel(f, 'Data set HEK-lenti', header=1, index_col='Target number').rename(get_key, axis='columns')
hct_df = pd.read_excel(f, 'Data set HCT-plasmid', header=1, index_col='Target number').rename(get_key, axis='columns')
hek_df = pd.read_excel(f, 'Data set HEK-plasmid', header=1, index_col='Target number').rename(get_key, axis='columns')
lenti_df['dataset'] = 'HEK-lenti'
hek_df['dataset'] = 'HEK-plasmid'
hct_df['dataset'] = 'HCT-plasmid'


## Train model on mini-human data
train_df = pd.concat([new_df.loc[new_df.chromatin_1000 == 0]])
X_train = pd.concat([process_sequence(train_df['sequence'])], axis=1)
y_train = train_df['efficiency']
rf_regressor = RandomForestRegressor(
    n_estimators=450,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
predictor = rf_regressor
predictor.fit(X_train, y_train)


## Predict sgRNA efficiencies
test_df = pd.concat([hek_df, hct_df, lenti_df])
test_df = test_df.loc[test_df.chromatin == 1]
X_test = pd.concat([process_sequence(test_df['sequence'])], axis=1)
y_test = test_df.efficiency
y_pred = predictor.predict(X_test)

## Plot predictions vs. truth
data = pd.concat([test_df.reset_index(), pd.DataFrame({'predicted': y_pred})], axis=1)
g = sns.lmplot(data=data, x='predicted', y='efficiency', hue='dataset')
g.set(xlim=(10,45),ylim=(0,100), xlabel='Predicted (%)', ylabel='Measured (%)', title='Random Forest model')
h = sns.lmplot(data=data, x='deepCPF', y='efficiency', hue='dataset')
h.set(xlim=(0,75),ylim=(0,100), xlabel='Predicted (%)', ylabel='Measured (%)', title='DeepCpf1 model')