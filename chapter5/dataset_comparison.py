import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind, kruskal

label_col = 'Indel frequency at endogenous target site (background subtracted)'

f = 'data/41587_2018_BFnbt4061_MOESM39_ESM.xlsx'

col_mappings = {
    'Indel frequency at endogenous target site (background subtracted)': 'efficiency',
    'Indel frequency at endogenous target site': 'efficiency',
    'Indel freqeuncy\n(Background substracted, %)': 'efficiency',
    'Indel frequency at synthetic target site (background subtracted)': 'syn_efficiency',
    'Chromatin accessibility (1= DNase I hypersensitive sites, 0 = Dnase I non-sensitive sites)': 'chromatin'
}

def get_key(i):
    return col_mappings[i] if i in col_mappings else i

lenti_df = pd.read_excel(f, 'Data set HEK-lenti', header=1, index_col='Target number').rename(get_key, axis='columns')
hct_df = pd.read_excel(f, 'Data set HCT-plasmid', header=1, index_col='Target number').rename(get_key, axis='columns')
hek_df = pd.read_excel(f, 'Data set HEK-plasmid', header=1, index_col='Target number').rename(get_key, axis='columns')
ht1_1_df = pd.read_excel(f, 'Data set HT 1-1', header=1).rename(get_key, axis='columns')


data = [
    {'name': 'HT 1-1', 'type': 'synthetic', 'values': ht1_1_df.efficiency},
    {'name': 'HEK-lenti', 'type': 'synthetic', 'values': lenti_df[lenti_df.chromatin == 1].syn_efficiency.reset_index(drop=True)},
    {'name': 'HEK-plasmid', 'type': 'accessible', 'values': hek_df[hek_df.chromatin == 1].efficiency.reset_index(drop=True)},
    {'name': 'HEK-lenti', 'type': 'accessible', 'values': lenti_df[lenti_df.chromatin == 1].efficiency.reset_index(drop=True)},
    {'name': 'HCT-plasmid', 'type': 'accessible', 'values': hct_df[hct_df.chromatin == 1].efficiency.reset_index(drop=True)},
    {'name': 'HEK-plasmid', 'type': 'inaccessible', 'values': hek_df[hek_df.chromatin == 0].efficiency.reset_index(drop=True)},
    {'name': 'HEK-lenti', 'type': 'inaccessible', 'values': lenti_df[lenti_df.chromatin == 0].efficiency.reset_index(drop=True)},
    {'name': 'HCT-plasmid', 'type': 'inaccessible', 'values': hct_df[hct_df.chromatin == 0].efficiency.reset_index(drop=True)}
]

data = pd.concat([pd.DataFrame(i['values'])
                  .assign(name=f'{i["name"]} ({i["type"]})')
                  .assign(target_type=i["type"])
                  .rename(columns={"syn_efficiency": "efficiency"}) for i in data]
                ).reset_index(drop=True)

###
### Create the box and whisker and density plots
###
sns.set(font_scale=1.4)
fig, ax = plt.subplots(1,2,figsize=(16,7))
g = sns.boxplot(data=data, x='efficiency', y='name', orient='h', ax=ax[0])
h = sns.kdeplot(data=data, x='efficiency', hue='name', common_norm=False, ax=ax[1])
g.set(xlim=(0, 100), xlabel='Efficiency (%)', title='Boxplot of the different datasets', ylabel='Name' )
h.set(xlim=(0, 100),ylim=(0, 0.05), xlabel='Efficiency (%)', title='Density plot of the different datasets')
fig.show()

###
### Uncomment to perform t-test between two groups
###
#Lenti - synthetic vs accessible (4.201553695904255 5.851801766032974e-05)
#t2, p2 = ttest_ind(data[data.name == 'HEK-lenti (synthetic)'].efficiency, data[data.name == 'HEK-lenti (accessible)'].efficiency)

#Synthetic vs synthetic (1.6433629573592865 0.10032875110538055)
#t2, p2 = ttest_ind(data[data.name == 'HT 1-1 (synthetic)'].efficiency, data[data.name == 'HEK-lenti (synthetic)'].efficiency)

#Synthetic vs accessible (14.72023643329492 2.851553749842077e-26)
#t2, p2 = ttest_ind(data[data.target_type == 'synthetic'].efficiency, data[data.target_type == 'accessible'].efficiency, equal_var = False)

#Synthetic vs endogynous (43.05856461387712 4.282011107220509e-139)
#t2, p2 = ttest_ind(data[data.target_type == 'synthetic'].efficiency, pd.concat([data[data.target_type == 'accessible'], data[data.target_type == 'inaccessible']]).efficiency, equal_var = False)

#Accessible vs inaccessible (7.235079350870042 8.924704693012026e-11)
#t2, p2 = ttest_ind(data[data.target_type == 'accessible'].efficiency, data[data.target_type == 'inaccessible'].efficiency, equal_var = False)

#Synthetic vs accessible (14.72023643329492 2.851553749842077e-26)
#t2, p2 = ttest_ind(data[data.target_type == 'synthetic'].efficiency, data[data.target_type == 'accessible'].efficiency, equal_var = False)

#HEK - delivery method (1.7921751683996383 0.08241695891472325)
#t2, p2 = ttest_ind(data[data.name == 'HEK-plasmid (accessible)'].efficiency, data[data.name == 'HEK-lenti (accessible)'].efficiency, equal_var = False)

#HEK - cell type (3.2832669179059235 0.002999441766760571)
t2, p2 = ttest_ind(data[data.name == 'HEK-plasmid (accessible)'].efficiency, data[data.name == 'HCT-plasmid (accessible)'].efficiency, equal_var = False)

print(t2, p2)


