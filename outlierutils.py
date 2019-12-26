import requests
import json

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score



class LabelSubmitter():
    def __init__(self, username, password, url='http://127.0.0.1:5000'):
        self.username = username
        self.password = password
        self.jwt_token = None
        self.base_url = url
        self.last_labels = None
        self.all_labels = None

    def get_jwt_token(self):
        """ Posts to /auth
        """
        auth = requests.post(self.base_url + '/auth', json={"username": f"{self.username}",
                "password": f"{self.password}"})
        try:
            self.jwt_token = json.loads(auth.text)['access_token']
        except KeyError:
            return auth

    def post_predictions(self, idx):
        """ Posts to /label
        sets self.last_labels
        """
        idx = [int(n) for n in idx] # replace numpy array and int64 by list with ints
        res = requests.post(url=self.base_url + '/label',
                    json={'data': {'idx': idx}},
                   headers={'Authorization': 'JWT {}'.format(self.jwt_token)})
        try:
            result = json.loads(res.text)['result']
            unzips = list(zip(*result))
            labels = pd.Series(index=unzips[0], data=unzips[1]).sort_index()
            self.last_labels = labels

            print(json.loads(res.text)['info'])
            print('number of positives in submission: {:d}'.format(int(labels.sum())))
            print('precision of submission: {:.2%}'.format(labels.mean()))
        except Exception:
            print(json.loads(res.text)['info'])

    def get_labels(self):
        """ 'Gets' to /label
        sets self.all_labels
        """
        try:
            res = requests.get(url=self.base_url + '/label',
                       headers={'Authorization': 'JWT {}'.format(self.jwt_token)})
            result = json.loads(res.text)['result']
            unzips = list(zip(*result))
            labels = pd.Series(index=unzips[0], data=unzips[1]).sort_index()
            self.all_labels = labels
            print('number of predictions made: {:d}'.format(int(len(labels))))
            print('total number of positives found: {:d}'.format(int(labels.sum())))
            print('total precision: {:.2%}'.format(labels.mean()))
        except Exception:
            print(json.loads(res.text))

    def get_statistics(self, plot=True):
        res = requests.get(url=self.base_url + '/labelstats',
           headers={'Authorization': 'JWT {}'.format(self.jwt_token)})
        stats = json.loads(res.text)['result']
        stats_df = pd.DataFrame.from_dict(stats).T
        stats_df['precision'] = 100 * stats_df['N_positives_found'] / stats_df['N_submitted']
        if plot:
            fig, axs = plt.subplots(2, 1, figsize=(12,6))
            stats_df['N_submitted'].plot(kind='bar', ax=axs[0])
            stats_df['precision'].plot(kind='bar', ax=axs[1])
            axs[0].set_title('Number of submitted points')
            axs[1].set_title('Precision [%]')
            plt.tight_layout()
        return stats_df

def plot_outlier_scores(scores):
    roc_score = roc_auc_score(train.isFraud, scores)
    classify_results = pd.DataFrame(data=pd.concat((train.isFraud, pd.Series(scores)), axis=1))
    classify_results.rename(columns={0:'score'}, inplace=True)
    sns.kdeplot(classify_results.loc[classify_results.isFraud==0, 'score'], label='negatives', shade=True, bw=0.01)
    sns.kdeplot(classify_results.loc[classify_results.isFraud==1, 'score'], label='positives', shade=True, bw=0.01)
    plt.title('AUC: {:.3f}'.format(roc_score))
    plt.xlabel('Score');
    return classify_results


def plot_top_N(scores, N=100):
    N = min(N, len(scores))
    classify_results = pd.DataFrame(data=pd.concat((train.isFraud, pd.Series(scores)), axis=1))
    classify_results.rename(columns={0:'score'}, inplace=True)
    classify_results = classify_results.sort_values(by='score', ascending=False)[:N]
    Npos_in_N = classify_results['isFraud'].sum()

    fig, ax = plt.subplots(1, 1, figsize=(16, 2))
    ims = ax.imshow(np.reshape(classify_results.isFraud.values, [1, -1]), extent=[-0.5, N, N/50, -0.5])
    ax.yaxis.set_visible(False)
    # ax.xaxis.set_ticklabels
    plt.colorbar(ims)
    plt.xlabel('Outlier rank [-]')
    plt.title(f'Number of positives found: {Npos_in_N} (P@Rank{N}: {Npos_in_N/N:.1%})')
    #plt.show()
    return classify_results

def median_imputation(df, median_impute_limit=0.95, impute_val=-999):
    """ inf/nan Values that occur more often than median_impute_limit are imputed with the median
    when less often, they are imputed by impute_val.
    Set median_impute_limit to 0 to always do median imputation
    """
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.columns:
        if not df[col].dtype == 'object':
            mean_nan = df[col].isna().mean()
            if mean_nan > median_impute_limit: # then, impute by median
                df[col] = df[col].fillna(df[col].median())
            elif mean_nan > 0 and mean_nan <= median_impute_limit:
                df[col] = df[col].fillna(impute_val)

    return df


def train_test_isoF(X_train, y_train, X_test=None, y_test=None, max_samples=1024, feature_list=None):
    if not feature_list is None:
        X_train, X_test = X_train[feature_list], X_test[feature_list]
    ifo = IsolationForest(n_estimators=50, max_samples=max_samples)
    ifo.fit(X_train)
    y_pred_ifo = ifo.decision_function(X_train)
    print('AUC Score on Train: {:.3f}'.format(roc_auc_score(y_train, -y_pred_ifo)))
    if X_test is None:
        return ifo
    y_pred_ifo_test = ifo.decision_function(X_test)
    print('AUC Score on Test: {:.3f}'.format(roc_auc_score(y_test, -y_pred_ifo_test)))
    return ifo
