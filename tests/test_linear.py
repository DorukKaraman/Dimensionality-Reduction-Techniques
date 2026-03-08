import numpy as np
import sys
sys.path.insert(0, '..')

from src.svd import TruncatedSVD, PCA, compress_image


def test_reconstruction_shape():
    # Reconstructed matrix should have same shape as original.
    A = np.random.randn(100, 80)
    svd = TruncatedSVD(n_components=10)
    svd.fit(A)
    assert svd.reconstruct().shape == A.shape


def test_orthogonality():
    # U should have orthonormal columns.
    A = np.random.randn(50, 30)
    svd = TruncatedSVD(n_components=10)
    svd.fit(A)
    
    UtU = svd.U_.T @ svd.U_
    assert np.allclose(UtU, np.eye(10), atol=1e-10)


def test_low_rank_exact():
    # Exact recovery of truly low-rank matrix.
    U = np.random.randn(50, 5)
    V = np.random.randn(5, 30)
    A = U @ V
    
    svd = TruncatedSVD(n_components=5)
    svd.fit(A)
    
    assert np.allclose(A, svd.reconstruct(), atol=1e-10)


def test_compress_image():
    # Basic image compression test.
    image = np.random.rand(64, 64) * 255
    compressed = compress_image(image, k=10)
    
    assert compressed.shape == image.shape
    assert compressed.min() >= 0
    assert compressed.max() <= 255


def test_transform_inverse_transform():
    # Transform then inverse_transform should give reconstruction.
    A = np.random.randn(50, 30)
    svd = TruncatedSVD(n_components=10)
    svd.fit(A)
    
    reduced = svd.transform(A)
    reconstructed = svd.inverse_transform(reduced)
    
    assert reduced.shape == (50, 10)
    assert np.allclose(reconstructed, svd.reconstruct(), atol=1e-10)


def test_fit_transform():
    # fit_transform should equal fit then U*s.
    A = np.random.randn(50, 30)
    svd = TruncatedSVD(n_components=10)
    
    result = svd.fit_transform(A)
    expected = svd.U_ * svd.s_
    
    assert np.allclose(result, expected, atol=1e-10)


def test_explained_variance_ratio():
    # Explained variance ratio should be positive and sum to <= 1.
    A = np.random.randn(50, 30)
    svd = TruncatedSVD(n_components=10)
    svd.fit(A)
    
    assert np.all(svd.explained_variance_ratio_ > 0)
    assert svd.explained_variance_ratio_.sum() <= 1.0
    assert np.all(np.diff(svd.explained_variance_ratio_) <= 0)


def test_invalid_n_components():
    # Should raise ValueError for invalid n_components.
    try:
        TruncatedSVD(n_components=0)
        assert False, "Should have raised"
    except ValueError:
        pass


def test_unfitted_raises():
    # Should raise RuntimeError when not fitted.
    svd = TruncatedSVD(n_components=5)
    try:
        svd.reconstruct()
        assert False, "Should have raised"
    except RuntimeError:
        pass


# PCA Tests

def test_pca_centering():
    # PCA should center data (mean subtracted).
    X = np.random.randn(100, 20) + 5  # Non-zero mean
    pca = PCA(n_components=10)
    pca.fit(X)
    
    assert np.allclose(pca.mean_, np.mean(X, axis=0))


def test_pca_transform_inverse():
    # Inverse transform should recover original (with enough components).
    X = np.random.randn(50, 20)
    pca = PCA(n_components=20)  # All components
    pca.fit(X)
    
    X_reduced = pca.transform(X)
    X_recovered = pca.inverse_transform(X_reduced)
    
    assert np.allclose(X, X_recovered, atol=1e-10)


def test_pca_explained_variance():
    # Variance should be positive and in descending order.
    X = np.random.randn(100, 30)
    pca = PCA(n_components=10)
    pca.fit(X)
    
    assert np.all(pca.explained_variance_ > 0)
    assert np.all(np.diff(pca.explained_variance_) <= 0)
    assert pca.explained_variance_ratio_.sum() <= 1.0


def test_pca_components_orthonormal():
    # Components should be orthonormal.
    X = np.random.randn(100, 50)
    pca = PCA(n_components=20)
    pca.fit(X)
    
    VVt = pca.components_ @ pca.components_.T
    assert np.allclose(VVt, np.eye(20), atol=1e-10)


def test_pca_none_components():
    # n_components=None should keep all.
    X = np.random.randn(50, 30)
    pca = PCA(n_components=None)
    pca.fit(X)
    
    assert pca.n_components_ == 30


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
    test_pca_centering()
    test_pca_transform_inverse()
    test_pca_explained_variance()
    test_pca_components_orthonormal()
    test_pca_none_components()
    print("All linear tests passed!")
