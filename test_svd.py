import numpy as np
from svd import TruncatedSVD, compress_image


def test_reconstruction_shape():
    # Reconstructed matrix should have same shape as original
    A = np.random.randn(100, 80)
    svd = TruncatedSVD(n_components=10)
    svd.fit(A)
    assert svd.reconstruct().shape == A.shape


def test_orthogonality():
    # U should have orthonormal columns
    A = np.random.randn(50, 30)
    svd = TruncatedSVD(n_components=10)
    svd.fit(A)
    
    UtU = svd.U_.T @ svd.U_
    assert np.allclose(UtU, np.eye(10), atol=1e-10)


def test_low_rank_exact():
    # Exact recovery of truly low-rank matrix
    U = np.random.randn(50, 5)
    V = np.random.randn(5, 30)
    A = U @ V
    
    svd = TruncatedSVD(n_components=5)
    svd.fit(A)
    
    assert np.allclose(A, svd.reconstruct(), atol=1e-10)


def test_compress_image():
    # Basic image compression test
    image = np.random.rand(64, 64) * 255
    compressed = compress_image(image, k=10)
    
    assert compressed.shape == image.shape
    assert compressed.min() >= 0
    assert compressed.max() <= 255


if __name__ == "__main__":
    test_reconstruction_shape()
    test_orthogonality()
    test_low_rank_exact()
    test_compress_image()
    print("All tests passed!")
