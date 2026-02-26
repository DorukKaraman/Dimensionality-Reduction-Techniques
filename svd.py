import numpy as np


class TruncatedSVD:
    # Truncated SVD for image compression
    
    def __init__(self, n_components):
        self.n_components = n_components
        self.U_ = None
        self.s_ = None
        self.Vt_ = None
    
    def fit(self, X):
        # Compute SVD and keep top k components
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        
        k = self.n_components
        self.U_ = U[:, :k]
        self.s_ = s[:k]
        self.Vt_ = Vt[:k, :]
        
        return self
    
    def reconstruct(self):
        # Return rank-k approximation
        return (self.U_ * self.s_) @ self.Vt_
    
    def compression_ratio(self, shape):
        # Calculate compression ratio: original_size / compressed_size
        m, n = shape
        k = self.n_components
        original = m * n
        compressed = k * (m + n + 1)
        return original / compressed


def compress_image(image, k):
    # Compress grayscale image using rank-k SVD approximation
    svd = TruncatedSVD(n_components=k)
    svd.fit(image)
    return np.clip(svd.reconstruct(), 0, 255)
