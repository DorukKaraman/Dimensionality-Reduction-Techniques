import numpy as np


def reconstruction_error(original, reconstructed, norm='fro'):
    """    
    Parameters
    ----------
    original : ndarray
        Original matrix.
    reconstructed : ndarray
        Reconstructed matrix.
    norm : str
        'fro' (Frobenius), 'mse', or 'mae'.
    
    Returns
    -------
    error : float
    """
    diff = original - reconstructed
    
    if norm == 'fro':
        return np.linalg.norm(diff, 'fro')
    elif norm == 'mse':
        return np.mean(diff ** 2)
    elif norm == 'mae':
        return np.mean(np.abs(diff))
    else:
        raise ValueError(f"Unknown norm: {norm}")


def relative_error(original, reconstructed):
    # Relative Frobenius error: ||A - A_k|| / ||A||
    return reconstruction_error(original, reconstructed) / np.linalg.norm(original, 'fro')


def psnr(original, reconstructed, max_val=255.0):
    # Peak Signal-to-Noise Ratio (dB). Higher is better.
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(max_val ** 2 / mse)
