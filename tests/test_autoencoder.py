import numpy as np
import sys
sys.path.insert(0, '..')

from src.autoencoder import LinearAutoencoder, ConvAutoencoder, TORCH_AVAILABLE


# Linear Autoencoder Tests

def test_autoencoder_shapes():
    # Test autoencoder input/output shapes.
    X = np.random.randn(50, 100)
    ae = LinearAutoencoder(n_components=10, n_iterations=10)
    ae.fit(X)
    
    latent = ae.transform(X)
    assert latent.shape == (50, 10)
    
    reconstructed = ae.inverse_transform(latent)
    assert reconstructed.shape == (50, 100)


def test_autoencoder_reconstruction():
    # Autoencoder should reduce loss over training.
    X = np.random.randn(100, 50)
    ae = LinearAutoencoder(n_components=20, learning_rate=0.1, 
                           n_iterations=200, random_state=42)
    ae.fit(X)
    
    assert ae.loss_history_[-1] < ae.loss_history_[0]


def test_autoencoder_centering():
    # Autoencoder should center data.
    X = np.random.randn(50, 30) + 10  # Non-zero mean
    ae = LinearAutoencoder(n_components=10, n_iterations=10)
    ae.fit(X)
    
    assert np.allclose(ae.mean_, np.mean(X, axis=0))


def test_autoencoder_unfitted_raises():
    # Should raise RuntimeError when not fitted.
    ae = LinearAutoencoder(n_components=5)
    try:
        ae.transform(np.random.randn(10, 20))
        assert False, "Should have raised"
    except RuntimeError:
        pass


# Convolutional Autoencoder Tests (PyTorch)

def test_conv_autoencoder_shapes():
    # Test conv autoencoder input/output shapes.
    if not TORCH_AVAILABLE:
        print("  Skipping (PyTorch not available)")
        return
    
    images = np.random.rand(20, 32, 32).astype(np.float32)
    conv_ae = ConvAutoencoder(n_filters=8, latent_channels=2, n_epochs=5, random_state=42)
    conv_ae.fit(images)
    
    reconstructed = conv_ae.reconstruct(images)
    assert reconstructed.shape == (20, 32, 32)
    
    latent = conv_ae.transform(images)
    assert latent.shape[0] == 20
    assert latent.shape[1] == 2  # latent_channels


def test_conv_autoencoder_loss_decreases():
    # Conv autoencoder should reduce loss over training.
    if not TORCH_AVAILABLE:
        print("  Skipping (PyTorch not available)")
        return
    
    images = np.random.rand(30, 32, 32).astype(np.float32)
    conv_ae = ConvAutoencoder(n_filters=8, latent_channels=2, n_epochs=20, random_state=42)
    conv_ae.fit(images)
    
    assert conv_ae.loss_history_[-1] < conv_ae.loss_history_[0]


def test_conv_autoencoder_filters():
    # Test that we can retrieve encoder filters.
    if not TORCH_AVAILABLE:
        print("  Skipping (PyTorch not available)")
        return
    
    images = np.random.rand(20, 32, 32).astype(np.float32)
    conv_ae = ConvAutoencoder(n_filters=16, latent_channels=4, n_epochs=5, random_state=42)
    conv_ae.fit(images)
    
    filters = conv_ae.get_encoder_filters()
    assert filters.shape[0] == 16  # n_filters
    assert filters.shape[1] == 3   # kernel height
    assert filters.shape[2] == 3   # kernel width


def test_conv_autoencoder_unfitted_raises():
    # Should raise RuntimeError when not fitted.
    if not TORCH_AVAILABLE:
        print("  Skipping (PyTorch not available)")
        return
    
    conv_ae = ConvAutoencoder(n_filters=8, latent_channels=2)
    try:
        conv_ae.transform(np.random.rand(10, 32, 32).astype(np.float32))
        assert False, "Should have raised"
    except RuntimeError:
        pass


if __name__ == "__main__":
    print("Running Linear Autoencoder tests...")
    test_autoencoder_shapes()
    test_autoencoder_reconstruction()
    test_autoencoder_centering()
    test_autoencoder_unfitted_raises()
    
    print("Running ConvAutoencoder tests...")
    test_conv_autoencoder_shapes()
    test_conv_autoencoder_loss_decreases()
    test_conv_autoencoder_filters()
    test_conv_autoencoder_unfitted_raises()
    
    print("All autoencoder tests passed!")
