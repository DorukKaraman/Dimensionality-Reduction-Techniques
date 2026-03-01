# Dimensionality Reduction Techniques for Images

SVD and PCA implementations for image compression and dimensionality reduction.

## Quick Start

```bash
pip install -r requirements.txt
python test_svd.py
```

## Demo

```bash
# SVD compression on single image
python visualize_compression.py

# PCA on synthetic face dataset  
python visualize_pca.py
```

**visualize_compression.py** - Shows SVD compression at different ranks with PSNR/error metrics.

**visualize_pca.py** - Demonstrates PCA on a dataset of synthetic faces, showing principal components and reconstructions.

## Project Structure

```
├── svd.py                    # TruncatedSVD and PCA classes
├── metrics.py                # Reconstruction error, PSNR metrics
├── visualize_compression.py  # SVD demo on single image
├── visualize_pca.py          # PCA demo on synthetic faces
├── test_svd.py               # Unit tests (15 tests)
├── requirements.txt
└── README.md
```

## Usage

```python
from svd import TruncatedSVD, PCA, compress_image
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
- [ ] Image utilities
- [ ] Jupyter notebook examples
- [ ] Autoencoder comparison

## License

MIT
