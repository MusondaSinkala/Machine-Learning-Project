"""
-------------------------------------------------------
creates the splits & directories for tensorflow data
-------------------------------------------------------
Author:  einsteinoyewole
Email:   eo2233@nyu.edu
__updated__ = "4/22/24"
-------------------------------------------------------
"""


# Imports
import dask.dataframe as dd
import dask.array as da
from dask.distributed import Client, LocalCluster
import os
import shutil
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit

# Constants
RANDOM_SEED=4
BASE_PATH = os.path.join(os.getcwd(), "Data")
# BASE_PATH = os.path.join(Path(os.getcwd()).resolve().parents[1], "Data")
assert os.path.isdir(BASE_PATH), f'Data Directory is required: {BASE_PATH}'
DATA_DIR = '/Users/einsteinoyewole/breast-histopathology-images/'  #TODO: Need to update this to where the images are unzipped
LOCAL = True

# derive the training, validation, and testing directories
TRAIN_PATH = os.path.sep.join([BASE_PATH, "training"])
VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
TEST_PATH = os.path.sep.join([BASE_PATH, "testing"])

def load_data():
    """
    -------------------------------------------------------
    loads the image directory and labels using dask dataframe
    (overloads with pandas)
    -------------------------------------------------------
    Returns:
       X: path file for each image (pathlike)
       y: label for corresponding image (int)
       group: Patient Id of the image (int)
    -------------------------------------------------------
    """
    print('Loading Data')
    assert os.path.isfile(os.path.join(BASE_PATH,'histopathological_data.csv')), 'data file is required'
    Y_col = 'cancer'
    group_col = 'PatientID'
    df = dd.read_csv(os.path.join(BASE_PATH,'histopathological_data.csv'))
    df['directory_path'] = df['directory_path'].str.replace('/content/breast-histopathology-images/',
                                                            DATA_DIR)
    X = df['directory_path'].str.cat(df['files'], sep=os.sep).to_dask_array(lengths=True)
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
       X: path file for each image (pathlike)
       y: label for corresponding image (int)
       group: Patient Id of the image (int)
    Returns:
       datasets: for each dataset a tuple of
            datasetname, array of image paths, array of labels, & tf path
    -------------------------------------------------------
    """
    def split_once(X, y, group):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
        train_ind, test_ind = next(gss.split(X, y, groups=group))
        X_train, X_test, y_train, y_test = X[train_ind], X[test_ind], y[train_ind], y[test_ind]
        group_train, group_test = group[train_ind], group[test_ind]
        return X_train, X_test, y_train, y_test, group_train, group_test

    print('Splitting train and test Data')
    X_train, X_test, y_train, y_test, group, _ = split_once(X, y, group)
    print('Splitting train and val Data')
    X_train, X_val, y_train, y_val, _, _ = split_once(X_train, y_train, group)
    return [
        ("training", X_train, y_train, TRAIN_PATH),
        ("validation", X_val, y_val, VAL_PATH),
        ("testing", X_test, y_test, TEST_PATH)
    ]

def create_tf_data_dir(datasets):
    """
    -------------------------------------------------------
    Copies images into a directory structure for tensorflow data
    -------------------------------------------------------
    Parameters:
       datasets: tuple containing each datasets data points
    -------------------------------------------------------
    """
    # loop over the datasets
    for (dType, imagePaths, labels, baseOutput) in datasets:

        # show which data split we are creating
        print("[INFO] building '{}' split".format(dType))
        # if the output base output directory does not exist, create it
        if not os.path.exists(baseOutput):
            print("[INFO] 'creating {}' directory".format(baseOutput))
            os.makedirs(baseOutput)
        # loop over the input image paths
        imagePaths, labels = imagePaths.compute(), labels.compute()
        for inputPath, label in tqdm(zip(imagePaths, labels)):
            filename = inputPath.split(os.path.sep)[-1]
            # build the path to the label directory
            labelPath = os.path.sep.join([baseOutput, str(label)])
            # if the label output directory does not exist, create it
            if not os.path.exists(labelPath):
                print("[INFO] 'creating {}' directory".format(labelPath))
                os.makedirs(labelPath)
            # construct the path to the destination image and then copy
            # the image itself
            p = os.path.sep.join([labelPath, filename])
            shutil.copy2(inputPath, p)


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
    create_tf_data_dir(datasets)
