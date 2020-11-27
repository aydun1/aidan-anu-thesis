import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import re

from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.utils import compute_class_weight
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_curve, mean_squared_error

import statsmodels.api as sm
import statsmodels.formula.api as smf

LENGTH_OF_PAM = 3
rev = {'A':'T', 'T':'A', 'C':'G', 'G':'C', 'a':'t', 't':'a', 'c':'g', 'g':'c'}
nucleotides = ['A', 'T', 'C', 'G']
dinucleotides = [i + j for i in nucleotides for j in nucleotides]
allnucleotides = nucleotides + dinucleotides
input_file = 'data/Gaetan.featureTable.xlsx'
model_file = 'cune_classifier.joblib'
RANDOM_STATE = 3767
N_TREES = 1000
RANDOM_STATE = 353563
FOLD_COUNT = 5
ARM_LENGTH = 65

def read_data(file_name):
    df = pd.read_excel(file_name, index_col='Id')
    df.WU_CRISPR.fillna(0, inplace=True)
    df = df[pd.notnull(df['ss_oligo'])]
    df.shape
    return df


def reverse_complement(sequence):
    return ''.join([rev[n] for n in sequence][::-1])


def gc_content(seq):
    d = len([s for s in seq if s in 'CcGc']) / len(seq) * 100
    return round(d, 2)


def get_oligo_components(guide, oligo, distance_from_pam, mutation_size, truncation=0):
    pam_mutation_index = range(20 - distance_from_pam - mutation_size, 20 - distance_from_pam)
    guide_regex = ''.join([nucleotide if i not in pam_mutation_index else '[ATCG]' for i, nucleotide in enumerate(guide)])
    match = re.search(guide_regex, oligo, flags=re.IGNORECASE)
    if match:
        dist = match.span()[1] - distance_from_pam - LENGTH_OF_PAM
    else:
        match = re.search(guide_regex, reverse_complement(oligo), flags=re.IGNORECASE)
        dist = len(oligo) - match.span()[1] + distance_from_pam + LENGTH_OF_PAM + 1
    mutation = oligo[dist-1: dist].upper()
    
    arms = [oligo[: dist-1].upper(), oligo[dist:].upper()]

    t_fix = [truncation] * 2 if truncation else [len(arm) for arm in arms]
  
    return pd.Series({
        'oligo_arm1': arms[0][-t_fix[0]:-20],
        'oligo_arm2': arms[1][20: t_fix[1]],       
        'oligo_arm1_length': len(arms[0]),
        'oligo_arm2_length': len(arms[1]),
        'oligo_trimmed': arms[0][-t_fix[0]:] + mutation + arms[1][: t_fix[1]]   
    })

   
def merge_duplicates(df, merge_key):
    tmp_df = df.reset_index().groupby(merge_key, group_keys=False).agg(
        {'Id': 'first',
         'point_mutations':sum,
         'indels': sum,
         'trials': sum}).set_index('Id')
    cols_to_use = df.columns.difference(tmp_df.columns)
    merged_df = pd.concat([df[cols_to_use], tmp_df], axis=1, join='inner')
    return merged_df


def remove_unwanted(df, vaa):
    #df3 = df[ ~(df.point_mutations / df.trials).between(range_from, range_to, True)]
    return df[ (df.indels / df.trials) > vaa]


def remove_dodgy(df):
    #df = df[df.trials > 1]
    return df[df.dodgy == 0]


def process_labels(df):
    labels_df = df.loc[:, ['indels', 'point_mutations', 'trials', 'New']]

    labels_df.loc[:, 'cleavage_sample_ratio'] = (labels_df.indels / labels_df.trials)
    for i in [0.4, 0.5, 0.6]:
        labels_df.loc[:, 'cleavage_sample_{}'.format(i)] = labels_df.cleavage_sample_ratio > i

    labels_df.loc[:, 'hdr_sample_ratio'] = (labels_df.point_mutations / labels_df.trials)
    for i in [0.1, 0.18, 0.2, 0.3, 0.4, 0.5, 0.6]:
        labels_df.loc[:, 'hdr_sample_{}'.format(i)] = labels_df.hdr_sample_ratio > i

    labels_df.loc[:, 'hdr_cleavage_ratio'] = (labels_df.point_mutations / labels_df.indels)
    for i in [0.1, 0.18, 0.2, 0.3, 0.4, 0.5, 0.6]:
        labels_df.loc[:, 'hdr_cleavage_{}'.format(i)] = labels_df.hdr_cleavage_ratio > i

    #labels_df.loc[:, 'hdr'] = (labels_df.point_mutations / labels_df.trials) > ((range_to + range_from) / 2)
    #labels_df.loc[:, 'hdr'] = (labels_df.point_mutations > 0)

    labels_df=labels_df.reset_index().set_index(['New','Id']).sort_index()
    return labels_df
          

def tokenize_sequence_global(seq, name):
    data = {n: int(sum(1 for i in range(len(seq)) if seq.startswith(n, i))/(len(seq)-len(n)+1)*100) for n in allnucleotides}
    s = pd.Series(data)
    s = s.reindex(allnucleotides, axis=1)
    s.index = ['{}_{}'.format(name, x) for x in s.index]
    return s


def tokenize_sequence_local(seq, name):   
    nmers = [1, 2]
    data = {'{}_{}_{:02d}'.format(name, n, i + 1):  seq[i: i + n] for n in nmers for i in range(0, len(seq) - n + 1)}
    return pd.Series(data)

def tokenize_sequence_local_onehot(seq, name):
    tokens = nucleotides + dinucleotides
    data = {f'loc{len(n)}_{i}_{n}': 1  if seq.startswith(n, i) else 0 for i in range(len(seq)) for n in tokens}
    s = pd.Series(data).astype('int32')
    s.index = ['{}_{}'.format(name, x) for x in s.index]
    return s

def process_features(df, guide_column_name, truncation):
    guide_spacer_df = df[guide_column_name].apply(lambda _: pd.Series(_[:-3], index=['spacer']))
    guide_local_df = guide_spacer_df.apply(lambda _: tokenize_sequence_local_onehot(_.spacer, 'guide'), axis=1)
    guide_global_df = guide_spacer_df.apply(lambda _: tokenize_sequence_global(_.spacer, 'guide'), axis=1)
    guide_N_df = df[guide_column_name].apply(lambda _: pd.Series(nucleotides.index(_[-3:-2]), index=['guide_N']))
    #guide_adjacent_df = df.o1.apply(lambda _: pd.Series(_, index=['guide_adjacent']))
    guide_gc = guide_spacer_df.apply(lambda _: gc_content(_.spacer), axis=1).rename('guide_gc_content')
   
    oligo_arms_df = df.apply(lambda _: get_oligo_components(_[guide_column_name], _.ss_oligo, _.distance_from_pam, _.mutation_size, truncation), axis=1)
    #oligo_arm1_local_df = oligo_arms_df.apply(lambda _: tokenize_sequence_local_onehot(_.oligo_arm1[::-1], 'arm1'), axis=1)
    #oligo_arm2_local_df = oligo_arms_df.apply(lambda _: tokenize_sequence_local_onehot(_.oligo_arm2, 'arm2'), axis=1).fillna(0)
    #oligo_all_global_df = oligo_arms_df.apply(lambda _: tokenize_sequence_global(_.oligo_trimmed, 'all'), axis=1)
    #oligo_arm1_global_df = oligo_arms_df.apply(lambda _: tokenize_sequence_global(_.oligo_arm1, 'arm1'), axis=1)
    #oligo_arm2_global_df = oligo_arms_df.apply(lambda _: tokenize_sequence_global(_.oligo_arm2, 'arm2'), axis=1)
    oligo_arm1_length = oligo_arms_df.apply(lambda _: _.oligo_arm1_length, axis=1).rename('arm1_length')
    oligo_arm2_length = oligo_arms_df.apply(lambda _: _.oligo_arm2_length, axis=1).rename('arm2_length')
    
    #oligo_arm1_gc = oligo_arms_df.apply(lambda _: gc_content(_.oligo_arm1), axis=1).rename('arm1_gc_content')
    #oligo_arm2_gc = oligo_arms_df.apply(lambda _: gc_content(_.oligo_arm2), axis=1).rename('arm2_gc_content')
    oligo_all_gc = oligo_arms_df.apply(lambda _: gc_content(_.oligo_trimmed), axis=1).rename('all_gc_content')

    
    # Original
    #features = [guide_global_df, guide_N_df, oligo_arm2_global_df]
    #features = [guide_local_df]
    #features = [df.distance_from_pam, oligo_arm1_length, oligo_arm2_length, guide_global_df, oligo_all_global_df,
    #            oligo_arm1_global_df, oligo_arm2_global_df, guide_gc, guide_local_df, oligo_arm2_local_df,
    #            oligo_all_gc, oligo_arm1_gc, oligo_arm2_gc]
    
    features= [guide_global_df, guide_local_df]
    
    #features = [df.distance_from_pam, oligo_arm2_length, guide_global_df, oligo_arm2_global_df, guide_local_df]
    
    #features = [oligo_arm1_length, oligo_arm1_global_df]
    #features = [oligo_arm2_length, oligo_arm2_global_df]

    #features = [guide_global_df, guide_N_df]
    #features = [guide_global_df, guide_local_df]
    #features = [guide_global_df, df.distance_from_pam]
    features = [oligo_arm1_length, oligo_arm2_length] #delete
    
    #features = [oligo_arm2_global_df]
    
    #features = [guide_global_df, guide_N_df, oligo_arm2_global_df, df.distance_from_pam]

    #features = [df.distance_from_pam]
    

    features_df = pd.concat(features, axis=1)
    
    #features_df.loc[:, 'New'] = df.New
    #features_df = features_df.reset_index().set_index(['New','Id']).sort_index()
    return features_df

def get_weights(y):
    return dict(zip(np.unique(y), compute_class_weight('balanced', classes=np.unique(y), y=y)))
    

def train_forest(X, y):
    class_weights = get_weights(y)
    forest = RandomForestClassifier(
        n_estimators=N_TREES,
        #min_samples_split=10,
        #max_features="log2",
        #oob_score=True,
        class_weight = class_weights,
        random_state=RANDOM_STATE
    )
    forest.fit(X, y)
    return forest



### Published dataset
df = read_data(input_file)
df = df[df.apply(lambda _: _.New in [0, 1, 2], axis=1)]
df = merge_duplicates(df, 'ss_oligo')
df = remove_unwanted(df, 0.4)
df = remove_dodgy(df)
features = process_features(df, 'full_guide_sequence', ARM_LENGTH)
#labels = process_labels(df)

X_train_original = features.loc[df.apply(lambda _: _.New in [0, 1, 2], axis=1), :]
y_train_original = labels.loc[[0, 1, 2], 'hdr_cleavage_0.4']

print('High:', sum(y_train_original))
print('Low:', len(y_train_original) - sum(y_train_original))
print('Hdr to cleavage:', np.median(labels.hdr_cleavage_ratio))
print('Hdr to sample ratio:', np.median(labels.hdr_sample_ratio))
print('Cleavage to sample ratio:', np.median(labels.cleavage_sample_ratio))


###V1 dataset
df = read_data(input_file)
df = df[df.apply(lambda _: _.New in [0, 1, 2, 3, 4, 5, 6], axis=1)]
df = merge_duplicates(df, 'ss_oligo')
df = remove_unwanted(df, 0.4)
df = remove_dodgy(df)
features = process_features(df, 'full_guide_sequence', ARM_LENGTH)
labels = process_labels(df)

X_test_original = features.loc[df.apply(lambda _: _.New in [3, 4, 5, 6], axis=1), :]
y_test_original = labels.loc[[3, 4, 5, 6], 'hdr_cleavage_0.4']
print('High:', sum(y_test_original))
print('Low:', len(y_test_original) - sum(y_test_original))
print('Hdr to cleavage:', np.median(labels.hdr_cleavage_ratio))
print('Hdr to sample ratio:', np.median(labels.hdr_sample_ratio))
print('Cleavage to sample ratio:', np.median(labels.cleavage_sample_ratio))


###Validate published model on V1
print('High:', sum(y_test_original))
print('Low:', len(y_test_original) - sum(y_test_original))

rf_out = train_forest(X_train_original, y_train_original).predict(X_test_original)
print(confusion_matrix(rf_out, y_test_original))
print(precision_score(rf_out, y_test_original))
print(recall_score(rf_out, y_test_original))
print(accuracy_score(rf_out, y_test_original))


###Validate published model on V2
df = read_data(input_file)
df = merge_duplicates(df, 'ss_oligo')
df = remove_unwanted(df, 0.4)
df = remove_dodgy(df)
features = process_features(df, 'full_guide_sequence', ARM_LENGTH)
labels = process_labels(df)
X_test_new = features.loc[df.apply(lambda _: _.New in [7], axis=1), :]
y_test_new = labels.loc[[7], 'hdr_cleavage_0.4']
print('High:', sum(y_test_new))
print('Low:', len(y_test_new) - sum(y_test_new))
print('Hdr to cleavage:', np.median(labels.hdr_cleavage_ratio))
print('Hdr to sample ratio:', np.median(labels.hdr_sample_ratio))
print('Cleavage to sample ratio:', np.median(labels.cleavage_sample_ratio))
rf_out = train_forest(X_train_original, y_train_original).predict(X_test_new)
print(confusion_matrix(rf_out, y_test_new))
print(precision_score(rf_out, y_test_new))
print(recall_score(rf_out, y_test_new))
print(accuracy_score(rf_out, y_test_new))



### Train on published data and V1
df = read_data(input_file)
df = merge_duplicates(df, 'ss_oligo')
df = remove_unwanted(df, 0.4)
df = remove_dodgy(df)
features = process_features(df, 'full_guide_sequence', 70)
labels = process_labels(df)
X_train_new = features.loc[df.apply(lambda _: _.New in [0, 1, 2, 3, 4, 5, 6], axis=1), :]
y_train_new = labels.loc[[0, 1, 2, 3, 4, 5, 6], 'hdr_cleavage_0.3']
X_test_new = features.loc[df.apply(lambda _: _.New in [7], axis=1), :]
y_test_new = labels.loc[[7], 'hdr_cleavage_0.4']
print('High:', sum(y_train_new))
print('Low:', len(y_train_new) - sum(y_train_new))
print('Hdr to cleavage:', np.median(labels.hdr_cleavage_ratio))
print('Hdr to sample ratio:', np.median(labels.hdr_sample_ratio))
print('Cleavage to sample ratio:', np.median(labels.cleavage_sample_ratio))



### Test on V2
#X_test_neww = X_test_new.assign(distance_from_pam = -20)
print('High:', sum(y_test_new))
print('Low:', len(y_test_new) - sum(y_test_new))
rf = train_forest(X_train_new, y_train_new)
preds = rf.predict(X_test_new)
print(confusion_matrix(preds, y_test_new))
print(precision_score(preds, y_test_new))
print(recall_score(preds, y_test_new))
print(accuracy_score(preds, y_test_new))