import numpy as np


class TruncatedSVD:
    # Truncated SVD with sklearn-style API
    
    def __init__(self, n_components):
        if n_components < 1:
            raise ValueError("n_components must be >= 1")
        self.n_components = n_components
        self.U_ = None
        self.s_ = None
        self.Vt_ = None
        self.explained_variance_ratio_ = None
    
    def fit(self, X):
        # Compute SVD and keep top k components.
        X = np.asarray(X, dtype=np.float64)
        
        max_k = min(X.shape)
        if self.n_components > max_k:
            raise ValueError(f"n_components={self.n_components} > min(m,n)={max_k}")
        
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        
        k = self.n_components
        self.U_ = U[:, :k]
        self.s_ = s[:k]
        self.Vt_ = Vt[:k, :]
        
        # Explained variance ratio
        total_var = np.sum(s ** 2)
        self.explained_variance_ratio_ = (self.s_ ** 2) / total_var
        
        return self
    
    def transform(self, X):
        # Project X into reduced space.
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        return X @ self.Vt_.T
    
    def inverse_transform(self, X_reduced):
        # Reconstruct from reduced representation
        self._check_fitted()
        return X_reduced @ self.Vt_
    
    def fit_transform(self, X):
        # Fit and transform in one step
        self.fit(X)
        return self.U_ * self.s_
    
    def reconstruct(self):
        # Return rank-k approximation of fitted matrix
        self._check_fitted()
        return (self.U_ * self.s_) @ self.Vt_
    
    def compression_ratio(self, shape):
        # Calculate compression ratio: original_size / compressed_size
        m, n = shape
        k = self.n_components
        original = m * n
        compressed = k * (m + n + 1)
        return original / compressed
    
    def _check_fitted(self):
        if self.Vt_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")


class PCA:
    # n_components : int or None
    
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.components_ = None          # Principal axes (Vt)
        self.mean_ = None                # Feature means
        self.explained_variance_ = None  # Variance per component
        self.explained_variance_ratio_ = None
        self.n_components_ = None        # Actual number retained
    
    def fit(self, X):
        # Fit PCA on dataset X (n_samples, n_features)
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        
        # Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # SVD of centered data
        U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # Determine number of components
        max_k = min(n_samples, n_features)
        if self.n_components is None:
            k = max_k
        else:
            if self.n_components > max_k:
                raise ValueError(f"n_components={self.n_components} > max={max_k}")
            k = self.n_components
        
        self.n_components_ = k
        self.components_ = Vt[:k, :]
        
        # Explained variance (eigenvalues of covariance matrix)
        self.explained_variance_ = (s[:k] ** 2) / (n_samples - 1)
        total_var = np.sum(s ** 2) / (n_samples - 1)
        self.explained_variance_ratio_ = self.explained_variance_ / total_var
        
        return self
    
    def transform(self, X):
        # Project X onto principal components
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) @ self.components_.T
    
    def inverse_transform(self, X_transformed):
        # Reconstruct from reduced representation
        self._check_fitted()
        return X_transformed @ self.components_ + self.mean_
    
    def fit_transform(self, X):
        # Fit and transform in one step
        self.fit(X)
        return self.transform(X)
    
    def _check_fitted(self):
        if self.components_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")


def compress_image(image, k):
    # Compress grayscale image using rank-k SVD approximation
    svd = TruncatedSVD(n_components=k)
    svd.fit(image)
    return np.clip(svd.reconstruct(), 0, 255)
