"""
Data preparation for Pendigits data.
The result of this script is input for the workshop participants.

This dataset has only numerical data (16 columns), with little meaning (originating from
downsampling coordinates in time from digits written on a digital pad)

Done here:
- mapping of outliers: b'yes'/b'no' to 1/0
- shuffling of data

Necessary preparation during the workshop:
- Nothing
"""

import pandas as pd
from outlierutils import reduce_mem_usage
from scipy.io import arff


## Path definitions
X_PATH = 'data/x_kdd.pkl'
Y_PATH = 'data/y_kdd.pkl'
kddcup_path = r'data/KDDCup99_original.arff'


## Load data
data = arff.loadarff(kddcup_path)
df = pd.DataFrame(data[0])
df = df.drop(columns=['id'])
df.outlier = df.outlier.map({b"'yes'":1, b"'no'":0})
df = df.sample(frac=1, random_state=2718)
df = df.reset_index(drop=True)


## Pickle the output
df.drop(columns='outlier').to_pickle(X_PATH)
df.outlier.to_pickle(Y_PATH)

print('Written output to: {}'.format(pendigits_path))
