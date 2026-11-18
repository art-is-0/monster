# Imports
import numpy as np

# ------------------------------------------------------------------------------

def error_rate(y_true, y_pred):
    return np.mean(y_true != y_pred)

# ------------------------------------------------------------------------------

def minimum_error(X_train, y_train):
    params = {}
    for i in np.unique(y_train):
        # Elements needed to solve for minimum error
        Xi = X_train[y_train == i]
        mu_i = Xi.mean(axis=0)
        sig_i = np.atleast_2d(np.cov(Xi, rowvar=False, bias=True))
        sig_i_inv = np.linalg.inv(sig_i)
        prior_i = np.mean(y_train == i)

        # Solving for Wi, wi, and wi0
        W = -(1/2) * sig_i_inv
        w = sig_i_inv @ mu_i
        w0 = -(1/2) * (mu_i @ sig_i_inv @ mu_i) - (1/2) * np.linalg.slogdet(sig_i)[1] + np.log(prior_i)

        params[i] = (W, w, w0)
    return params


# def discriminant(x, W, w, w0):
#     return (x @ W @ x) + (w @ x) + w0

def discriminant(X, W, w, w0):
    quad = np.einsum('ni,ij,nj->n', X, W, X)  # x^T W x for every row
    return quad + X @ w + w0


# def minimum_error_predict(params, X_test):
#     y_pred = []
#     lst = [0]*len(params)
#     for x in X_test:
#         for key in params.keys():
#             lst[key] = discriminant(x, *params[key])

#         y_pred.append(np.argmax(lst))
#     return np.array(y_pred)

def minimum_error_predict(params, X_test):
    labels = sorted(params)
    scores = np.stack([discriminant(X_test, *params[k]) for k in labels], axis=1)
    return np.array(labels)[scores.argmax(axis=1)]

# ------------------------------------------------------------------------------

def augment(X):
    n = X.shape[0]
    column1 = np.ones(n)
    return np.column_stack([column1, X])


def least_squares(X_train, y_train):
    Y = augment(X_train)
    b = y_train.copy()
    b[b == 2] = -1

    a = np.linalg.inv(Y.T @ Y) @ Y.T @ b
    return a


def least_squares_predict(a, X_test):
    Y_test = augment(X_test)
    y_pred = []
    for y in Y_test:
        g = a @ y
        y_pred.append(1 if g >= 0 else 2)
    return np.array(y_pred)

# ------------------------------------------------------------------------------

def nearest_neighbor(X_train, y_train, X_test):
    y_pred = []
    for x in X_test:
        dists = np.linalg.norm(X_train - x, axis=1)
        idx_near = np.argmin(dists)
        y_pred.append(y_train[idx_near])
    return np.array(y_pred)