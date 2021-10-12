import numpy as np

def compute_mse(y, tx, w):
    """
    mse
    """
    return np.sum((y - np.dot(tx, w.T))**2) / (2 * tx.shape[0])

