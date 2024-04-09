import numpy as np

def real_uvrms(data):
    """Compute the real root mean square."""
    return np.sqrt(np.mean(data ** 2, axis=1)) * 1e6

def fake_uvrms(data):
    """Compute the 'fake RMS', which is the standard deviation.
       This is how OpenBCI GUI computes it
    """
    return np.std(data, axis=1) * 1e6