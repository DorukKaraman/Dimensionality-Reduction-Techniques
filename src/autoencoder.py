import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


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


class ConvAutoencoder:
    """
    Convolutional autoencoder for image compression using PyTorch.
    
    Architecture:
        Input (H, W) → Conv → ReLU → Conv → Latent → ConvT → ReLU → ConvT → Output (H, W)
    
    Parameters
    ----------
    n_filters : int, default=16
        Number of filters in convolutional layers.
    latent_channels : int, default=4
        Number of channels in the bottleneck (latent) layer.
    learning_rate : float, default=0.001
        Learning rate for Adam optimizer.
    n_epochs : int, default=50
        Number of training epochs.
    batch_size : int, default=16
        Mini-batch size for training.
    random_state : int, optional
        Random seed for reproducibility.
    """
    
    def __init__(self, n_filters=16, latent_channels=4, learning_rate=0.001, 
                 n_epochs=50, batch_size=16, random_state=None):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for ConvAutoencoder. Install with: pip install torch")
        
        self.n_filters = n_filters
        self.latent_channels = latent_channels
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        
        self.model_ = None
        self.loss_history_ = []
        self.image_shape_ = None
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _build_model(self, H, W):
        """Build encoder-decoder CNN."""
        
        class _ConvAE(nn.Module):
            def __init__(self, n_filters, latent_channels):
                super().__init__()
                
                # Encoder: (1, H, W) -> (n_filters, H/2, W/2) -> (latent, H/4, W/4)
                self.encoder = nn.Sequential(
                    nn.Conv2d(1, n_filters, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(n_filters, latent_channels, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                )
                
                # Decoder: (latent, H/4, W/4) -> (n_filters, H/2, W/2) -> (1, H, W)
                self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(latent_channels, n_filters, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(n_filters, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.Sigmoid(),  # Output in [0, 1]
                )
            
            def forward(self, x):
                latent = self.encoder(x)
                return self.decoder(latent)
            
            def encode(self, x):
                return self.encoder(x)
            
            def decode(self, z):
                return self.decoder(z)
        
        return _ConvAE(self.n_filters, self.latent_channels)
    
    def fit(self, images, verbose=False):
        """
        Train the convolutional autoencoder.
        
        Parameters
        ----------
        images : ndarray of shape (n_samples, H, W)
            Grayscale images normalized to [0, 1].
        verbose : bool, default=False
            Print loss every epoch.
        
        Returns
        -------
        self
        """
        images = np.asarray(images, dtype=np.float32)
        n_samples, H, W = images.shape
        self.image_shape_ = (H, W)
        
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
        
        # Build model
        self.model_ = self._build_model(H, W).to(self.device_)
        
        # Prepare data: (N, H, W) -> (N, 1, H, W)
        X_tensor = torch.from_numpy(images[:, np.newaxis, :, :]).to(self.device_)
        
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        self.loss_history_ = []
        
        for epoch in range(self.n_epochs):
            # Shuffle indices
            indices = np.random.permutation(n_samples)
            epoch_loss = 0.0
            n_batches = 0
            
            for start_idx in range(0, n_samples, self.batch_size):
                batch_idx = indices[start_idx:start_idx + self.batch_size]
                batch = X_tensor[batch_idx]
                
                optimizer.zero_grad()
                output = self.model_(batch)
                loss = criterion(output, batch)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            self.loss_history_.append(avg_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.n_epochs}, Loss: {avg_loss:.6f}")
        
        return self
    
    def transform(self, images):
        """
        Encode images to latent representation.
        
        Parameters
        ----------
        images : ndarray of shape (n_samples, H, W)
        
        Returns
        -------
        latent : ndarray of shape (n_samples, latent_channels, H/4, W/4)
        """
        self._check_fitted()
        images = np.asarray(images, dtype=np.float32)
        X_tensor = torch.from_numpy(images[:, np.newaxis, :, :]).to(self.device_)
        
        self.model_.eval()
        with torch.no_grad():
            latent = self.model_.encode(X_tensor)
        
        return latent.cpu().numpy()
    
    def inverse_transform(self, latent):
        """
        Decode latent representation to images.
        
        Parameters
        ----------
        latent : ndarray of shape (n_samples, latent_channels, H/4, W/4)
        
        Returns
        -------
        images : ndarray of shape (n_samples, H, W)
        """
        self._check_fitted()
        z_tensor = torch.from_numpy(latent.astype(np.float32)).to(self.device_)
        
        self.model_.eval()
        with torch.no_grad():
            output = self.model_.decode(z_tensor)
        
        return output.cpu().numpy()[:, 0, :, :]  # Remove channel dim
    
    def reconstruct(self, images):
        """Encode and decode images."""
        self._check_fitted()
        images = np.asarray(images, dtype=np.float32)
        X_tensor = torch.from_numpy(images[:, np.newaxis, :, :]).to(self.device_)
        
        self.model_.eval()
        with torch.no_grad():
            output = self.model_(X_tensor)
        
        return output.cpu().numpy()[:, 0, :, :]
    
    def get_encoder_filters(self):
        """
        Get the learned convolutional filters from the first encoder layer.
        
        Returns
        -------
        filters : ndarray of shape (n_filters, kernel_h, kernel_w)
        """
        self._check_fitted()
        # First conv layer weights: (out_channels, in_channels, kH, kW)
        weights = self.model_.encoder[0].weight.data.cpu().numpy()
        return weights[:, 0, :, :]  # (n_filters, kH, kW)
    
    def _check_fitted(self):
        if self.model_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
