import numpy as np
import sys
sys.path.insert(0, '..')

from src.metrics import reconstruction_error, relative_error, psnr


def test_reconstruction_error_zero():
    # Reconstruction error should be zero for identical matrices.
    A = np.random.randn(50, 30)
    assert reconstruction_error(A, A, 'fro') == 0
    assert reconstruction_error(A, A, 'mse') == 0
    assert reconstruction_error(A, A, 'mae') == 0


def test_reconstruction_error_positive():
    # Reconstruction error should be positive for different matrices.
    A = np.random.randn(50, 30)
    B = A + np.random.randn(50, 30) * 0.1
    
    assert reconstruction_error(A, B, 'fro') > 0
    assert reconstruction_error(A, B, 'mse') > 0
    assert reconstruction_error(A, B, 'mae') > 0


def test_relative_error_zero():
    # Relative error should be zero for identical matrices.
    A = np.random.randn(50, 30)
    assert relative_error(A, A) == 0


def test_relative_error_bounded():
    # Relative error should be bounded between 0 and 1 for small perturbations.
    A = np.random.randn(50, 30)
    B = A + np.random.randn(50, 30) * 0.1
    
    err = relative_error(A, B)
    assert 0 < err < 1


def test_psnr_infinite():
    # PSNR should be infinite for identical matrices.
    A = np.random.randn(50, 30)
    assert psnr(A, A) == float('inf')


def test_psnr_positive():
    # PSNR should be positive for different matrices.
    A = np.random.randn(50, 30)
    B = A + np.random.randn(50, 30) * 0.1
    
    assert psnr(A, B) > 0


def test_psnr_higher_for_similar():
    # PSNR should be higher for more similar matrices.
    A = np.random.randn(50, 30)
    B_close = A + np.random.randn(50, 30) * 0.01
    B_far = A + np.random.randn(50, 30) * 0.5
    
    assert psnr(A, B_close) > psnr(A, B_far)


if __name__ == "__main__":
    test_reconstruction_error_zero()
    test_reconstruction_error_positive()
    test_relative_error_zero()
    test_relative_error_bounded()
    test_psnr_infinite()
    test_psnr_positive()
    test_psnr_higher_for_similar()
    print("All metrics tests passed!")
