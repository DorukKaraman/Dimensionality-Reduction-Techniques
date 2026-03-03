import numpy as np


class LinearAutoencoder:
    """
    Linear autoencoder for dimensionality reduction.
    
    Architecture:
        Input (n_features) → Encoder → Latent (n_components) → Decoder → Output (n_features)
    
    Parameters
    ----------
    n_components : int
        Dimension of the latent (compressed) representation.
    learning_rate : float, default=0.01
        Learning rate for gradient descent.
    n_iterations : int, default=1000
        Number of training iterations.
    random_state : int, optional
        Random seed for reproducibility.
    """
    
    def __init__(self, n_components, learning_rate=0.01, n_iterations=1000, random_state=None):
        self.n_components = n_components
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state
        
        self.W_encoder_ = None  # Shape: (n_features, n_components)
        self.W_decoder_ = None  # Shape: (n_components, n_features)
        self.mean_ = None
        self.loss_history_ = []
    
    def fit(self, X, verbose=False):
        """
        Train the autoencoder on data X.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        verbose : bool, default=False
            Print loss every 100 iterations.
        
        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        
        # Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Initialize weights (Xavier initialization)
        rng = np.random.default_rng(self.random_state)
        scale = np.sqrt(2.0 / (n_features + self.n_components))
        self.W_encoder_ = rng.normal(0, scale, (n_features, self.n_components))
        self.W_decoder_ = rng.normal(0, scale, (self.n_components, n_features))
        
        self.loss_history_ = []
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Forward pass
            latent = X_centered @ self.W_encoder_  # (n_samples, n_components)
            reconstruction = latent @ self.W_decoder_  # (n_samples, n_features)
            
            # Loss: MSE
            error = reconstruction - X_centered
            loss = np.mean(error ** 2)
            self.loss_history_.append(loss)
            
            if verbose and (i + 1) % 100 == 0:
                print(f"Iteration {i+1}/{self.n_iterations}, Loss: {loss:.6f}")
            
            # Backward pass (gradients)
            grad_decoder = (latent.T @ error) / n_samples  # (n_components, n_features)
            grad_encoder = (X_centered.T @ (error @ self.W_decoder_.T)) / n_samples  # (n_features, n_components)
            
            # Update weights
            self.W_encoder_ -= self.learning_rate * grad_encoder
            self.W_decoder_ -= self.learning_rate * grad_decoder
        
        return self
    
    def transform(self, X):
        """
        Encode X into latent representation.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        
        Returns
        -------
        latent : ndarray of shape (n_samples, n_components)
        """
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) @ self.W_encoder_
    
    def inverse_transform(self, latent):
        """
        Decode latent representation back to original space.
        
        Parameters
        ----------
        latent : ndarray of shape (n_samples, n_components)
        
        Returns
        -------
        X_reconstructed : ndarray of shape (n_samples, n_features)
        """
        self._check_fitted()
        return latent @ self.W_decoder_ + self.mean_
    
    def fit_transform(self, X, verbose=False):
        """Fit and transform in one step."""
        self.fit(X, verbose=verbose)
        return self.transform(X)
    
    def reconstruct(self, X):
        """Encode and decode X."""
        return self.inverse_transform(self.transform(X))
    
    def _check_fitted(self):
        if self.W_encoder_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
