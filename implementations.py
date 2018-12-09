
# coding: utf-8

# # Part 1

# ### Error functions

import numpy as np
from imp_functions import sample_data

def mean_square_error(labels, data, weights):
    return 1 / (2 * len(data)) * np.nansum((labels - data.dot(weights))**2)

def mean_square_gradient(labels, data, weights):
    return -1 / len(data) * data.T.dot(labels - data.dot(weights))

def rmse(labels, data, weights):
    mse = mean_square_error(labels, data, weights)
    return math.sqrt(2 * mse)


def logistic_error(labels, data, weights):
    predictions = data.dot(weights)
    loss = np.sum(np.logaddexp(0, predictions)) - labels.T.dot(predictions)
    return np.squeeze(loss / labels.shape[0])

def logistic_gradient(labels, data, weights):
    pred = sigmoid(data.dot(weights))
    grad = data.T.dot(pred - labels)
    return grad / labels.shape[0]


def reg_logistic_gradient(y, tx, lambda_, w):
    """return the loss and gradient."""
    num_samples = y.shape[0]
    loss = logistic_error(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
    gradient = logistic_gradient(y, tx, w) + 2 * lambda_ * w
    return loss, gradient


# ### Mathematical functions

def sigmoid(t):
    return np.exp(-np.logaddexp(0, -t))


# ### Algorithms

# Least squares
# Optimized by gradient descent over max_iters iterations and learning rate gamma
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    
    w = initial_w
    
    for i in range(max_iters):
        gradient = mean_square_gradient(y, tx, w)
        w = w - gamma * gradient
        
    return w, mean_square_error(y, tx, w)



# Least squares
# Optimized by stochastic gradient descent over max_iters iterations and learning rate gamma
# Using a batch size of 1 for the stochastic gradient computation
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
        
    w = initial_w
    
    for i in range(max_iters):
        chosen_index = np.random.choice(np.arange(tx.shape[0]))
        gradient = mean_square_gradient(y[chosen_index], tx[chosen_index], w)
        w = w - gamma * gradient
        
    return w, mean_square_error(y, tx, w)

# Least squares
# Normal equations
def least_squares(y, tx):
    A = tx.T.dot(y)
    XtX = tx.T.dot(tx)
    w_optimal = np.linalg.solve(XtX, A)
    return w_optimal, mean_square_error(y, tx, w_optimal)

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    return w, mean_square_error(y, tx, w)

# Logistic regression
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    
    w = initial_w
    
    for i in range(max_iters):
        y, tx = sample_data(y, tx , seed = 7, size_samples=tx.shape[0])
        gradient = logistic_gradient(y, tx, w)
        w = w - gamma * gradient
        
    return w, logistic_error(y, tx, w)


# Regularized logistic regression with L2 norm
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    
    w = initial_w
    
    for i in range(max_iters):
        y, tx = sample_data(y, tx , seed = 7, size_samples=tx.shape[0])
        loss,gradient = reg_logistic_gradient(y, tx, lambda_, w)
        w = w - gamma * gradient
        
    return w, loss

