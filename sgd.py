import numpy as np
from batch_iter import batch_iter

def compute_mse(y, tx, w):

    return np.sum((y - np.dot(tx, w.T))**2) / (2 * tx.shape[0])


def compute_mse_gradient(y, tx, w):
    """Compute the gradient."""
    e = y - np.dot(tx, w.T)

    return -(np.dot(tx.T, e))/y.shape[0]

def compute_mse_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""

    e = y - np.dot(tx, w.T)

    return -(np.dot(tx.T, e))/y.shape[0]


def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_iters, gamma
        , compute_loss=compute_mse
        , compute_stoch_gradient=compute_mse_stoch_gradient):
    """Stochastic gradient descent algorithm."""

    losses = []
    ws = []
    w = initial_w
    for batch_y, batch_tx in batch_iter(y, tx, batch_size, num_batches=max_iters):
        loss = compute_loss(batch_y, batch_tx, w)
        grad = compute_stoch_gradient(batch_y, batch_tx, w)
        w = w - gamma * grad
        # store w and loss
        ws.append(w)
        losses.append(loss)

    return losses, ws
    

def gradient_descent(y, tx, initial_w, max_iters, gamma
    , compute_loss=compute_mse
    , compute_gradient=compute_mse_gradient):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):

        loss = compute_loss(y, tx, w)
        grad = compute_gradient(y, tx, w)

        w = w - gamma * grad
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws
