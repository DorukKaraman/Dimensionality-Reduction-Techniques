import numpy as np
from svd import TruncatedSVD, compress_image
from metrics import reconstruction_error, relative_error, psnr


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


def test_transform_inverse_transform():
    # Transform then inverse_transform should give reconstruction
    A = np.random.randn(50, 30)
    svd = TruncatedSVD(n_components=10)
    svd.fit(A)
    
    reduced = svd.transform(A)
    reconstructed = svd.inverse_transform(reduced)
    
    assert reduced.shape == (50, 10)
    assert np.allclose(reconstructed, svd.reconstruct(), atol=1e-10)


def test_fit_transform():
    # fit_transform should equal fit then U*s
    A = np.random.randn(50, 30)
    svd = TruncatedSVD(n_components=10)
    
    result = svd.fit_transform(A)
    expected = svd.U_ * svd.s_
    
    assert np.allclose(result, expected, atol=1e-10)


def test_explained_variance_ratio():
    A = np.random.randn(50, 30)
    svd = TruncatedSVD(n_components=10)
    svd.fit(A)
    
    # Should be positive and sum to <= 1
    assert np.all(svd.explained_variance_ratio_ > 0)
    assert svd.explained_variance_ratio_.sum() <= 1.0
    # Should be in descending order
    assert np.all(np.diff(svd.explained_variance_ratio_) <= 0)


def test_invalid_n_components():
    try:
        TruncatedSVD(n_components=0)
        assert False, "Should have raised"
    except ValueError:
        pass


def test_unfitted_raises():
    svd = TruncatedSVD(n_components=5)
    try:
        svd.reconstruct()
        assert False, "Should have raised"
    except RuntimeError:
        pass


def test_metrics():
    A = np.random.randn(50, 30)
    B = A + np.random.randn(50, 30) * 0.1
    
    # Basic sanity checks
    assert reconstruction_error(A, A, 'fro') == 0
    assert reconstruction_error(A, B, 'fro') > 0
    assert reconstruction_error(A, B, 'mse') > 0
    
    assert relative_error(A, A) == 0
    assert 0 < relative_error(A, B) < 1
    
    # PSNR
    assert psnr(A, A) == float('inf')
    assert psnr(A, B) > 0


if __name__ == "__main__":
    test_reconstruction_shape()
    test_orthogonality()
    test_low_rank_exact()
    test_compress_image()
    test_transform_inverse_transform()
    test_fit_transform()
    test_explained_variance_ratio()
    test_invalid_n_components()
    test_unfitted_raises()
    test_metrics()
    print("All tests passed!")
