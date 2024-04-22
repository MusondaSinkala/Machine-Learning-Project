"""
-------------------------------------------------------

-------------------------------------------------------
Author:  einsteinoyewole
Email:   eo2233@nyu.edu
__updated__ = "4/22/24"
-------------------------------------------------------
"""


# Imports
import os
import xgboost as xgb
import dask.dataframe as dd
import dask.array as da
from dask.distributed import Client, LocalCluster
from dask.diagnostics import ProgressBar
from dask_ml.model_selection import train_test_split, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import GroupShuffleSplit, HalvingGridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from scipy.stats import uniform, loguniform, randint
import numpy as np
import pandas as pd
import json

# Constants
LOCAL = True
RANDOM_SEED=4
BASE_PATH = os.getcwd()
# BASE_PATH = Path(os.getcwd()).resolve().parents[1]
Y_col = 'cancer'
group_col = 'PatientID'
X_cols = ['MPI', 'SDPI', 'OTV', 'LM', 'UPP', 'LPP', 'Xcoords', 'Ycoords']
# TODO: Remove the search json file to perform hyperparameter search again
hyperparams = {
    "reg_alpha": np.logspace(1e-5, 1e-1, num=3),  # L1 regularization term on weights.
    # "reg_lambda": np.logspace(1e-5, 1e-1, num=5), # L2 regularization term on weights
    # "gamma": np.linspace(1, 9, num=3), # Minimum loss reduction required to make a further partition on a leaf node of the tree
    'colsample_bytree': np.linspace(0.8,1.2, num=3),  #  subsample ratio of columns when constructing each tree
    "max_depth": np.arange(5,8,1), # Maximum depth of a tree
    "sampling_method":['uniform'],  #method to use to sample the training instances
    # "learning_rate": np.logspace(1e-5, 1e-1, num=5),
    'n_estimators': np.arange(100,200, 50),
    "verbosity": [0],
    "seed": [RANDOM_SEED],
    "objective": ["binary:logistic"]
}


class json_serialize(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.integer):
      return int(obj)
    if isinstance(obj, np.floating):
      return float(obj)
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    return json.JSONEncoder.default(self, obj)

def load_data():
    """
    -------------------------------------------------------
    loads the extracted image data and labels using dask dataframe
    (overloads with pandas)
    -------------------------------------------------------
    Returns:
       X: extracted independent columns from images (dask.array[X_cols])
       y: label for corresponding image (dask.array[int])
       group: Patient Id of the image (dask.array[int])
    -------------------------------------------------------
    """
    print('Loading Data')
    assert os.path.isfile(os.path.join(BASE_PATH, 'Data','histopathological_data.csv')), 'data file is required'
    df = dd.read_csv(os.path.join(BASE_PATH, 'Data', 'histopathological_data.csv'))
    X = df[X_cols].to_dask_array(lengths=True)
    y = df[Y_col].to_dask_array(lengths=True)
    group = df[group_col].to_dask_array(lengths=True)
    return X, y, group

def get_train_test_val_split(X, y, group):
    """
    -------------------------------------------------------
    Splits the data into train test and validation datasets based on each Patient
    Patient Id overlap can cause data leakage
    -------------------------------------------------------
    Parameters:
       X: extracted independent columns from images (dask.array[X_cols])
       y: label for corresponding image (dask.array[int])
       group: Patient Id of the image (dask.array[int])
    Returns:
       datasets: for each dataset a dictionary of
            datasetname -> array of image paths & array of labels
    -------------------------------------------------------
    """
    def split_once(X, y, group):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
        train_ind, test_ind = next(gss.split(X, y, groups=group))
        X_train, X_test, y_train, y_test = X[train_ind], X[test_ind], y[train_ind], y[test_ind]
        group_train, group_test = group[train_ind], group[test_ind]
        return X_train, X_test, y_train, y_test, group_train, group_test

    print('Splitting train and test Data')
    X_train, X_test, y_train, y_test, group, group_test = split_once(X, y, group)
    print('Splitting train and val Data')
    X_train, X_val, y_train, y_val, group_train, group_val = split_once(X_train, y_train, group)
    return {
        "training": (X_train, y_train, group_train),
        "validation": (X_val, y_val, group_val),
        "testing": (X_test, y_test, group_test)
    }
def hyperparameter_search(train_data, val_data):
    """
    -------------------------------------------------------
    Performs a halving grid search on the parameter space
    -------------------------------------------------------
    Parameters:
       train_data: X, y and groups data (tuple(dask.array, dask.array, dask.array))
       val_data: X, y and groups data (tuple(dask.array, dask.array, dask.array))
    Returns:
       df: result of hyperparameter search (pd.DataFrame)
    -------------------------------------------------------
    """
    # Read the written search result if search has been performed
    if os.path.isfile(os.path.join(BASE_PATH, 'results', 'xgboost_res.json')):
        print("Using old hyperparameter search result")
        with open(os.path.join(BASE_PATH, 'results', 'xgboost_res.json'), 'r') as fp:
            r = json.load(fp)
        return pd.DataFrame(r).sort_values(by='mean_test_score', ascending=False)

    gss = GroupShuffleSplit(n_splits=3, test_size=0.2, random_state=RANDOM_SEED)
    clf = HalvingGridSearchCV(xgb.XGBClassifier(tree_method='hist', n_jobs=-1), hyperparams,
                              scoring='accuracy',
                              random_state=RANDOM_SEED,
                              n_jobs=-1,
                              refit=False,
                              verbose=3,
                              cv=gss
                              )
    print('perfoming a new hyperparameter search')
    clf.fit(train_data[0], train_data[1], groups=train_data[2])
    # Write the search results
    with open(os.path.join(BASE_PATH, 'results', 'xgboost_res.json'), 'w') as fp:
        json.dump(clf.cv_results_, fp, cls=json_serialize)
    with open(os.path.join(BASE_PATH, 'results', 'xgboost_res.json'), 'r') as fp:
        r = json.load(fp)
    # Return search results as dataframe
    return pd.DataFrame(r).sort_values(by='mean_test_score', ascending=False)


def train_model(dask_client, train_data, val_data, parameters):
    """
    -------------------------------------------------------
    Fits an XGBoost model with given parameters using train data
    -------------------------------------------------------
    Parameters:
        dask_client: dask distributed client
       train_data: X, y and groups data (tuple(dask.array, dask.array, dask.array))
       val_data: X, y and groups data (tuple(dask.array, dask.array, dask.array))
    Returns:
       clf: (xgb.dask.DaskXGBClassifier)
    -------------------------------------------------------
    """
    early_stop = xgb.callback.EarlyStopping(
        rounds=5, metric_name='logloss', save_best=True
    )
    defualt_params = {'random_state':RANDOM_SEED, 'objective':'binary:logistic',
                                     'verbosity':3, 'callbacks':[early_stop]}
    clf = xgb.dask.DaskXGBClassifier(**{**defualt_params, **parameters})
    clf.client = dask_client
    clf.fit(train_data[0], train_data[1], eval_set=[(val_data[0], val_data[1])])
    return clf


if __name__ == "__main__":
    if LOCAL:
        cluster = LocalCluster()
        client = Client(cluster, asynchronous=False)
    else:
        from dask_mpi import initialize
        initialize(interface='ib0', local_directory=os.getenv('TMPDIR'))
        client = Client()
    X, y, group = load_data()
    datasets = get_train_test_val_split(X, y, group)
    results_df = hyperparameter_search(datasets['training'], datasets['validation'])
    best_params = results_df.loc[results_df['mean_test_score'].idxmax(), 'params']
    model = train_model(client, datasets['training'], datasets['validation'], best_params)
    X_test, y_test, _ = datasets['testing']
    y_pred = model.predict(X_test)
    print()
    print(classification_report(y_test, y_pred))


