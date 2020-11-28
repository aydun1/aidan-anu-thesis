import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import re


loxP_forward_re = re.compile('''(?P<flank5>^[ATCG]*)(?P<loxp>ATAACTTCGTATAG[ATCG]{6}TTATACGAAGTTAT)(?P<flank3>[ATCG]*$)''', flags=re.IGNORECASE)
loxP_reverse_re = re.compile('''(?P<flank5>^[ATCG]*)(?P<loxp>ATAACTTCGTATAA[ATCG]{6}CTATACGAAGTTAT)(?P<flank3>[ATCG]*$)''', flags=re.IGNORECASE)

def quantify_donor(row, prime):
    sequence = row[f'donor_{prime}prime'].rstrip()
    f_match = loxP_forward_re.match(sequence)
    r_match = loxP_reverse_re.match(sequence)
    match = f_match if f_match else r_match
    direction = 1 if f_match else 0
    if not match: return 0
    
    denominator = row[f'loxp_{prime}prime'] + row[f'indel_{prime}prime']
    efficiency = row[f'loxp_{prime}prime'] / denominator if denominator else 0
    
    return pd.Series({
        'guide': row[f'guide_{prime}prime'],
        'flank5_length': len(match.group('flank5')),
        'flank3_length': len(match.group('flank3')),
        'flank5': match.group('flank5')[-40:],
        'flank3': match.group('flank3')[:40],
        #'bigger_5': len(match.group('flank5')) > len(match.group('flank3')),
        #'total_length': len(match.group('flank5')) + len(match.group('flank3')),
        'efficiency': efficiency,
        'direction': direction,
        'distance': row['distance'],
        'concentration': row.ssodn_concentration_ng_ul,
        'live_born_pups': row['live_born_pups']
    })



floxing_df = pd.read_excel('data/chapter3.xlsx')
floxing_df = remove_dodgy(floxing_df)


#print('High:', sum(floxing_y))
#print('Low:', len(floxing_y) - sum(floxing_y))
#print('Hdr to cleavage:', np.median(labels.hdr_cleavage_ratio))
#print('Hdr to sample ratio:', np.median(labels.hdr_sample_ratio))
#print('Cleavage to sample ratio:', np.median(labels.cleavage_sample_ratio))

odf3 = floxing_df.apply(lambda _: quantify_donor(_, 3), axis=1)
odf5 = floxing_df.apply(lambda _: quantify_donor(_, 5), axis=1)

odf = pd.concat([odf3, odf5])
#odf = odf[odf.efficiency > 0]
#odf = odf[odf.efficiency < 1]

#odf = odf[odf.direction == 1]

X_5 = odf['flank5'].apply(lambda _: tokenize_sequence_global(_, 'oligo_5'))
X_3 = odf['flank3'].apply(lambda _: tokenize_sequence_global(_, 'oligo_3'))
X = pd.concat([X_5, X_3, odf.flank3_length, odf.flank5_length, odf.concentration, odf.live_born_pups], axis=1)
y = odf['efficiency'] > 0.4

print(sum(y) / len(y))

cv = KFold(FOLD_COUNT, shuffle=True)
folds = [fold for fold in cv.split(X, y)]

rf = RandomForestClassifier(
    n_estimators=1000,
    oob_score=True,
    random_state=RANDOM_STATE
)

combined = [[0, 0],[0, 0]]
oob_ave = 0
mse_ave = 0
precision_ave = 0
recall_ave = 0
accuracy_ave = 0
for fold in folds:
    rf.fit(X.iloc[fold[0]], y.iloc[fold[0]])
    #print(rf.oob_score_)
    preds = rf.predict(X.iloc[fold[1]])
    cf = confusion_matrix(preds, y.iloc[fold[1]])
    #combined += cf
    print(cf)
    oob_ave += rf.oob_score_
    #mse_ave += mean_squared_error(y.iloc[fold[1]], preds)
    #precision_ave += precision_score(preds, y.iloc[fold[1]])
    #recall_ave += recall_score(preds, y.iloc[fold[1]])
    #accuracy_ave += accuracy_score(preds, y.iloc[fold[1]])
    
print(combined)
#print('OOB:', oob_ave / FOLD_COUNT)
print('MSE:', mse_ave / FOLD_COUNT)

def quantify_donor_all(row):
    new_row = {}
    for p in [5, 3]:
        sequence = row[f'donor_5prime'].rstrip()
        f_match, r_match = loxP_forward_re.match(sequence), loxP_reverse_re.match(sequence)
        direction = 1 if f_match else 0
        match = f_match if f_match else r_match
        _denominator = row[f'loxp_{p}prime'] + row[f'indel_{p}prime']
        efficiency = row[f'loxp_{p}prime'] / _denominator if _denominator else 0
        new_row[f'ssodn{p}_flank5_length'] = len(match.group('flank5'))
        new_row[f'ssodn{p}_flank3_length'] = len(match.group('flank3'))
        new_row[f'ssodn{p}_flank5'] = match.group('flank5')[-40:]
        new_row[f'ssodn{p}_flank3'] = match.group('flank3')[:40]
        new_row[f'guide{p}'] = row[f'guide_{p}prime']
        new_row[f'efficiency{p}'] = efficiency
    new_row['efficiency'] = True if row.loxp_both else False
    new_row['concentration'] = row.ssodn_concentration_ng_ul
    new_row['live_born_pups'] = row.live_born_pups
    return pd.Series(new_row)
            
            
    #if not match_3 or match_5: return 0
    #'bigger_5': len(match.group('flank5')) > len(match.group('flank3')),
    #'total_length': len(match.group('flank5')) + len(match.group('flank3')),
    #'direction': direction,
    #'distance': row['distance'],



odf = floxing_df.apply(lambda _: quantify_donor_all(_), axis=1)

#odf = odf[odf.efficiency > 0]
#odf = odf[odf.efficiency < 1]

#odf = odf[odf.direction == 1]

X_5 = odf['ssodn5_flank5'].apply(lambda _: tokenize_sequence_global(_, 'o5_f5'))
X_3 = odf['ssodn5_flank3'].apply(lambda _: tokenize_sequence_global(_, 'o5_f3'))
X = pd.concat([odf.efficiency5, odf.efficiency3, X_5, X_3, odf.ssodn5_flank3_length, odf.ssodn5_flank5_length, odf.concentration, odf.live_born_pups], axis=1)
y = odf['efficiency']

print(sum(y) / len(y))

cv = KFold(FOLD_COUNT, shuffle=True)
folds = [fold for fold in cv.split(X, y)]

rf = RandomForestClassifier(
    n_estimators=1000,
    oob_score=True,
    random_state=RANDOM_STATE
)

combined = [[0, 0],[0, 0]]
oob_ave = 0
mse_ave = 0
precision_ave = 0
recall_ave = 0
accuracy_ave = 0
for fold in folds:
    rf.fit(X.iloc[fold[0]], y.iloc[fold[0]])
    #print(rf.oob_score_)
    preds = rf.predict(X.iloc[fold[1]])
    cf = confusion_matrix(preds, y.iloc[fold[1]])
    combined += cf
    print(cf)
    oob_ave += rf.oob_score_
    #mse_ave += mean_squared_error(y.iloc[fold[1]], preds)
    #precision_ave += precision_score(preds, y.iloc[fold[1]])
    #recall_ave += recall_score(preds, y.iloc[fold[1]])
    #accuracy_ave += accuracy_score(preds, y.iloc[fold[1]])
    
print(combined)
#print('OOB:', oob_ave / FOLD_COUNT)
print('MSE:', mse_ave / FOLD_COUNT)

for i in sorted(zip(X, rf.feature_importances_), key=lambda x: x[1], reverse=True):
    print(i)