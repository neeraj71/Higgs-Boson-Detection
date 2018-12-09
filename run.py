import numpy as np
import matplotlib.pyplot as plt
import datetime
import csv

## Ridge regression model
def mean_square_error(labels, data, weights):
    return 1 / (2 * len(data)) * np.nansum((labels - data.dot(weights))**2)

def ridge_regression(y, tx, lambda_):
    """
    Performs ridge regression
    :param y: labels
    :param tx: features
    :param lambda_: lambda parameter of the ridge regression
    :returns: (optimal weights, mean square error of the estimation)
    """

    # The matrices from least squares
    A = tx.T.dot(y)
    XtX = tx.T.dot(tx)
  
    # Adding the regularizer
    lambda_prime = lambda_ * 2 * tx.shape[0]
    XtXlI = XtX + lambda_prime * np.eye(XtX.shape[0])

    w_optimal = np.linalg.solve(XtXlI, A)
  
    return w_optimal, mean_square_error(y, tx, w_optimal)


## Data I/O
def load_csv_data(filename, sub_sample=False):
    """
    Loads data from a csv file
    :param filename: the full path to the data csv file
    :param sub_sample: return a smaller subset of the data
    :returns: y (class labels), tX (features) and ids (event ids)
    """

    y = np.genfromtxt(filename, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(filename, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    :param ids: event ids associated with each prediction
    :param y_pred: predicted class labels
    :param name: string name of .csv output file to be created
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})


## Data processing
def split_data(x, y, ratio, myseed=1):
    """
    Shuffle then split the dataset in 2 subsets based on the split ratio
    :param x: data features
    :param y: lables
    :param ratio: split ratio
    :param myseed: seed used for the shuffle randomizer
    :returns: training data, testing data, training labels, testing labels
    """
    # set seed
    np.random.seed(myseed)
    # generate random indices
    num_row = len(y)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[: index_split]
    index_te = indices[index_split:]
    # create split
    x_tr = x[index_tr]
    x_te = x[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]
    return x_tr, x_te, y_tr, y_te


def predict(x, w):
    """
    Given a dataset and some weights, return the classifier predictions
    :param x: data
    :param w: weights
    :returns: array of predictions: signals are classified as 1 and background as 0
    """
    y_pred = np.dot(x,w)
    y_pred[y_pred > 0] = 1
    y_pred[y_pred <= 0] = -1 
    return y_pred


def build_poly(x, degree):
    """
    Perform data augmentation using a polynomial basis of a specified degree
    :param x: original data
    :param degree: the degree of the polynomial basis used for extension
    :return: the augmented data
    """
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly,np.power(x, deg)]
    return poly


def remove_missing_data(mat):
    """
    Replace missing data entries (values -999 in the dataset) with the median
    for that feature
    :param mat: the data matrix to process
    :returns: the data matrix mat with missing values replaced with column median
    """
    for i in range(0,mat.shape[1]):
        col = [x for x in mat[:,i] if x!=-999]
        median = np.median(col)
        index = np.where(mat[:,i] == -999)
        mat[index,i] = median
    return mat



def normalize(mat, train_mean=None, train_std=None):
    """
    Perform data normalisation to a distribution of mean 0 and variance 1.
    The normalisation for test data is perfomed using the mean and standard
    deviation of the train data.
    :param mat: the data matrix
    :param train_mean: if None, the mean will be computed from mat;
                       otherwise, this mean will be used for normalisation
    :param train_std: if None, the standard deviation will be computed from mat;
                      otherwise, this value will be used for normalisation
    :returns: the normalized mat, the mean and the standard deviation used
              for normalisation
    """

    if train_mean is not None and train_std is not None:
        mat = (mat-train_mean)/train_std
        return mat

    mean = np.mean(mat, axis = 0)
    std = np.std(mat, axis = 0)
    mat = (mat - mean) / std
    return mat, mean, std



def get_group(y, x, group_number):
    """
    Split the data into groups, using the values of the column 22, PRI_JET_NUM,
    as follows:
        Group 0 -> jet num = 0
        Group 1 -> jet num = 1
        Group 2 -> jet num = 2,3
    Also, removes missing columns associated with each group.

    :param y: the data labels or data ids
    :param x: the data matrix
    :param group_number: the group id. Must be 1, 2 or 3
    :returns: data_labels/ids and the filtered data entries of the group group_number
    :raise ValueError: if the group_number is out of range
    """

    # Columns to remove for each grouup
    i_0 = [22, 4, 5, 6, 12, 23, 24, 25, 26, 27, 28, 29]
    i_1 = [22, 4, 5, 6, 12, 26, 27, 28]
    i_2 = [22]

    col_index = [i_0, i_1, i_2]

    # Select data entries that correspond to group group_number
    if group_number < 2 :
        ind = np.where(x[:,22] == group_number)
    else :
        ind = np.where((x[:,22] == group_number) | (x[:,22] == (group_number+1)))

    # Select data entries in the group group_number and
    # remove the corresponding columns
    data = x[ind]
    data_label = y[ind]
    data = np.delete(data,col_index[group_number],axis=1)

    return data_label, data


def training_ridge(y, x, degree=9, lambda_=0.000001):
    """
    Train a ridge regression based model for detecting Higgs boson decays.

    The data is split into 3 datasets, based on the PRI_JET_NUM column
    (column 22), as follows:
        Group 0 -> jet num = 0
        Group 1 -> jet num = 1
        Group 2 -> jet num = 2,3

    The PRI_JET_NUM column indicates what data is missing from the experiment,
    so we drop the corresponding columns and the column 22.

    Three different ridge regression models are trained on these groups and
    used for classification.

    :param y: data labels used for training
    :param x: data matrix
    :param degree: the degree of the polynomial basis used for data augmentation
    :param lambda_: the parameter lambda of the ridge regression
    """


    means          = []
    stds           = []
    weights        = []
    training_acc   = []
    validation_acc = []

    # The number of training and testing data points
    tr_num  = 0
    val_num = 0

    for group_number in range(3):

        data_label, data, = get_group(y, x, group_number)

        print('Group {} : {}'.format(group_number, data.shape))

        data = remove_missing_data(data)

        # Normalisation / data augmentation
        data, mean, std = normalize(data)
        data = build_poly(data, degree)

        x_tr, x_val, y_tr, y_val = split_data(data, data_label, ratio = 0.9, myseed = 7)

        tr_num = x_tr.shape[0] + tr_num
        val_num = x_val.shape[0] + val_num

        # Perform ridge regression
        weight, err = ridge_regression(y_tr, x_tr, lambda_)

        # Do training/testing evaluation
        y_tr_pred = predict(x_tr, weight)
        y_val_pred = predict(x_val, weight)

        # Find the number of correctly classified entries in the current group
        # and append them to a list to compute the total accuracy
        tr_acc = np.sum(y_tr_pred == y_tr)
        val_acc = np.sum(y_val_pred == y_val)

        training_acc.append(tr_acc)
        validation_acc.append(val_acc)

        # Get the mean and standard deviation for each group required to
        # normalize test data
        means.append(mean)
        stds.append(std)

        # Get the weights for each group
        weights.append(weight)


    # Compute and display total accuracy
    print('Training Accuracy : ', np.sum(training_acc) / tr_num)
    print('Validation Accuracy : ', np.sum(validation_acc) / val_num)

    # Return the optimal weights and the normalisation parameters of each group
    return weights, means, stds, degree


def test_data_predictions_ridge(x, id, means, stds, weights, degree):
    """
    Do predictions using the trained model using the above routine.
    :param x: the test data
    :param id: the ids of the entries in the test data
    :param means: the means used to normalize the 3 groups
    :param stds: the standard deviations used to normalize the 3 groups
    :param weights: the weights of the three models
    :param degree: the degree of the polynomial used for data augmentation
    """

    test_pred = []
    test_id = []
    for group_number in range(3):

        data_id, data, = get_group(id, x, group_number)

        data = remove_missing_data(data)
        data = normalize(data,means[group_number], stds[group_number])

        data = build_poly(data,degree)
        pred = predict(data, weights[group_number])

        test_pred.append(pred)                 # append the predictions for each group
        test_id.append(data_id)                # append the corresponding ids for each group

    test_id = np.concatenate(test_id,0)          # get a numpy array of ids
    test_pred = np.concatenate(test_pred,0)      # get a numpy array of corresponding predictions

    create_csv_submission(test_id,test_pred,'sb14') 


if __name__ == '__main__':

    y, x, ids = load_csv_data('data/train.csv')

    weights, means, stds, degree = training_ridge(y, x)

    _,x_test, ids = load_csv_data('data/test.csv')
    test_data_predictions_ridge(x_test, ids, means, stds, weights, degree)

