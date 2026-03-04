# Dimensionality Reduction Techniques for Images

SVD, PCA, and Autoencoder implementations for image compression and dimensionality reduction.

## Quick Start

```bash
pip install -r requirements.txt
python tests.py
```

## Demos

```bash
# SVD compression on single image
python visualize_compression.py

# PCA on synthetic face dataset  
python visualize_pca.py

# Linear Autoencoder training and learned features
python visualize_linear_ae.py

# Convolutional Autoencoder training and filters
python visualize_conv_ae.py

# Compare all methods: PCA vs Linear AE vs Conv AE
python compare_methods.py
```

## Project Structure

```
├── svd.py                     # TruncatedSVD and PCA classes
├── autoencoder.py             # LinearAutoencoder (NumPy) + ConvAutoencoder (PyTorch)
├── metrics.py                 # Reconstruction error, PSNR metrics
├── visualize_compression.py   # SVD demo on single image
├── visualize_pca.py           # PCA demo on synthetic faces
├── visualize_linear_ae.py     # Linear Autoencoder demo + weights
├── visualize_conv_ae.py       # Conv Autoencoder demo + filters
├── compare_methods.py         # PCA vs Linear AE vs Conv AE comparison
├── tests.py                   # Unit tests (23 tests)
├── requirements.txt
└── README.md
```

## Usage

```python
from svd import TruncatedSVD, PCA, compress_image
from autoencoder import LinearAutoencoder
from metrics import psnr, relative_error
import numpy as np

# SVD: Compress a single image
image = np.random.rand(256, 256) * 255
compressed = compress_image(image, k=50)
print(f"PSNR: {psnr(image, compressed):.1f} dB")

# PCA: Reduce dimensionality of a dataset
# X shape: (n_samples, n_features) - each row is a flattened image
X = np.random.randn(100, 1024)  # 100 images, 32x32 pixels

pca = PCA(n_components=50)
X_reduced = pca.fit_transform(X)
print(f"Variance retained: {pca.explained_variance_ratio_.sum():.2%}")

X_reconstructed = pca.inverse_transform(X_reduced)

# Autoencoder: Neural network approach
linear_ae = LinearAutoencoder(n_components=50, learning_rate=0.1, n_iterations=500)
linear_ae.fit(X)
X_linear = linear_ae.reconstruct(X)

# Convolutional Autoencoder (PyTorch)
images = np.random.rand(100, 32, 32).astype(np.float32)  # (N, H, W)
from autoencoder import ConvAutoencoder
conv_ae = ConvAutoencoder(n_filters=16, latent_channels=4, n_epochs=50)
conv_ae.fit(images)
images_reconstructed = conv_ae.reconstruct(images)
```

## API Reference

### TruncatedSVD

| Method | Description |
|--------|-------------|
| `fit(X)` | Compute SVD of matrix X |
| `transform(X)` | Project X into reduced space |
| `inverse_transform(X_reduced)` | Reconstruct from reduced representation |
| `fit_transform(X)` | Fit and transform in one step |
| `reconstruct()` | Get rank-k approximation of fitted matrix |
| `compression_ratio(shape)` | Calculate storage compression ratio |

### PCA

| Method | Description |
|--------|-------------|
| `fit(X)` | Fit PCA on dataset X (centers data) |
| `transform(X)` | Project X onto principal components |
| `inverse_transform(X_reduced)` | Reconstruct from reduced representation |
| `fit_transform(X)` | Fit and transform in one step |

**Attributes:** `components_`, `mean_`, `explained_variance_`, `explained_variance_ratio_`

### LinearAutoencoder

| Method | Description |
|--------|-------------|
| `fit(X, verbose)` | Train autoencoder with gradient descent |
| `transform(X)` | Encode X to latent space |
| `inverse_transform(latent)` | Decode from latent space |
| `fit_transform(X)` | Fit and encode in one step |
| `reconstruct(X)` | Encode then decode X |

**Parameters:** `n_components`, `learning_rate`, `n_iterations`, `random_state`

**Attributes:** `W_encoder_`, `W_decoder_`, `mean_`, `loss_history_`

### ConvAutoencoder (PyTorch)

| Method | Description |
|--------|-------------|
| `fit(images, verbose)` | Train CNN autoencoder on images (N, H, W) |
| `transform(images)` | Encode images to latent feature maps |
| `inverse_transform(latent)` | Decode latent representation to images |
| `reconstruct(images)` | Encode then decode images |
| `get_encoder_filters()` | Get learned first-layer conv filters |

**Parameters:** `n_filters`, `latent_channels`, `learning_rate`, `n_epochs`, `batch_size`, `random_state`

**Attributes:** `model_`, `loss_history_`, `image_shape_`

### Metrics

| Function | Description |
|----------|-------------|
| `reconstruction_error(A, B, norm)` | Frobenius, MSE, or MAE error |
| `relative_error(A, B)` | Normalized Frobenius error |
| `psnr(A, B, max_val)` | Peak Signal-to-Noise Ratio (dB) |

## Roadmap

- [x] TruncatedSVD with sklearn-style API
- [x] PCA for datasets
- [x] Reconstruction error metrics
- [x] Linear Autoencoder (NumPy)
- [x] Convolutional Autoencoder (PyTorch)
- [x] Method comparison visualization
- [ ] Image utilities for loading, flattening and reshaping
- [ ] Jupyter notebook examples
- [ ] More datasets (e.g. MNIST, CIFAR-10)
- [ ] Refactoring code (structuring images etc and generating synthetic data organization)
## License

MIT
