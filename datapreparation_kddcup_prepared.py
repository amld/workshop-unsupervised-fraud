"""
Data preparation for KDD cup data.
The result of this script is input for the workshop participants.

This dataset has 3 categorical variables, and seems to give good results with outlier algorithms

Done here:
- transformation from byte-type strings to regular utf8 strings
- mapping of outliers: 'yes'/'no' to 1/0
- shuffling of data

Necessary preparation during the workshop:
- Removal of duplicates
(NB: df.duplicated(df.drop(columns=['id']).columns).sum() shows there are duplicates)
- (For some algorithms) normalization, categorical encoding

"""

import pandas as pd
from scipy.io import arff
import numpy as np

## Path definitions
X_PATH = 'data/x_kdd_prepared.pkl'
Y_PATH = 'data/y_kdd_prepared.pkl'

kddcup_path = r'data/KDDCup99_withoutdupl_norm_1ofn.arff'


## Load data
data = arff.loadarff(kddcup_path)

df = pd.DataFrame(data[0])

# Convert byte columns to regular strings
str_df_columns = df.select_dtypes([np.object]).columns
for col in str_df_columns:
    df[col] = df[col].str.decode('utf-8')
    df[col] = df[col].apply(lambda x: x.lstrip('\'').rstrip('\''))
df.outlier = df.outlier.map({'yes':1, 'no':0})

## Shuffle the columns
df = df.sample(frac=1, random_state=2718)
df = df.reset_index(drop=True)
df = df.drop(columns='id')

## Pickle the output
df.drop(columns='outlier').to_pickle(X_PATH)
df.outlier.to_pickle(Y_PATH)
print('Written output to: {}'.format(X_PATH))
