"""
Data preparation for IEEE Fraud dataset.
The result of this script is input for the workshop participants.

This dataset has many categorical variables and missing values, and is challenging (and presumably realistic)

Done here:
- Joining of two datasets: transaction and identity
- Adding a column (has_id) that indicates whether a matching identity record was present
- Selecting only first N rows (per default 200K)
- reducing the precision of the numbers to decrease memory usage
- saving the columns of the transaction dataset (may be useful at some point)

Necessary preparation during the workshop:
- normalization, categorical encoding, NaN filling, ....

NB: in Anaconda installation, execute with pythonw
"""

import pandas as pd
from outlierutils import reduce_mem_usage

## Settings
n_rows = int(200E3) # keep data set small enough (full is about 400E3)

## Path definitions
RAW_TRAX_DATA_PATH = 'bigdata/train_transaction.csv'
RAW_ID_DATA_PATH = 'bigdata/train_identity.csv'

X_DATA_PATH = 'bigdata/x_fraud.pkl'
Y_DATA_PATH = 'bigdata/y_fraud.pkl'

TRAXCOLUMNS_PATH = 'bigdata/trax_columns.pkl'

## Load data
# "Left table": load only n rows
data_transaction = pd.read_csv(RAW_TRAX_DATA_PATH, nrows=n_rows)

# "Right table": load all (to be sure nothing is missing for the join)
data_identity = pd.read_csv(RAW_ID_DATA_PATH)
data_identity['has_id'] = 1 # to identify those that had matching identity info

## Join data
train = data_transaction.merge(data_identity, on='TransactionID', how='left')
train['has_id'] = train['has_id'].fillna(0)

transaction_cols = pd.Series(data_transaction.columns)
del data_transaction, data_identity
train = reduce_mem_usage(train)


## Pickle the output
transaction_cols.to_pickle(TRAXCOLUMNS_PATH)
train.isFraud.to_pickle(Y_DATA_PATH) # y-labels
train.drop(columns=['isFraud']).to_pickle(X_DATA_PATH) # Data without y-labels

print('Data written to {}, {} and {}'.format(X_DATA_PATH,
                                            Y_DATA_PATH,
                                            TRAXCOLUMNS_PATH))
