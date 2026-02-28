# Dimensionality Reduction Techniques for Images

SVD-based image compression demonstrating the Eckart-Young-Mirsky theorem.

## Quick Start

```bash
pip install -r requirements.txt
python test_svd.py
```

## Demo

```bash
python visualize_compression.py
```

Generates a visualization comparing the original image with compressed versions at different ranks (5, 20, 50, 100), showing PSNR and relative error for each.

## Project Structure

```
├── svd.py                    # TruncatedSVD class with sklearn-style API
├── metrics.py                # Reconstruction error, PSNR metrics
├── visualize_compression.py  # Demo visualization script
├── test_svd.py               # Unit tests
├── requirements.txt
└── README.md
```

## Usage

```python
from svd import TruncatedSVD, compress_image
from metrics import reconstruction_error, psnr
import numpy as np

# Simple API
image = ...  # grayscale image (H, W)
compressed = compress_image(image, k=50)

# sklearn-style API
svd = TruncatedSVD(n_components=50)
svd.fit(image)
print(f"Explained variance: {svd.explained_variance_ratio_.sum():.2%}")

reduced = svd.transform(image)      # Project to reduced space
reconstructed = svd.inverse_transform(reduced)

# Quality metrics
print(f"PSNR: {psnr(image, reconstructed):.1f} dB")
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

### Metrics

| Function | Description |
|----------|-------------|
| `reconstruction_error(A, B, norm)` | Frobenius, MSE, or MAE error |
| `relative_error(A, B)` | Normalized Frobenius error |
| `psnr(A, B, max_val)` | Peak Signal-to-Noise Ratio (dB) |

## Roadmap

- [ ] Add PCA for face datasets (eigenfaces)
- [x] sklearn-style API (transform/inverse_transform)
- [x] Reconstruction error metrics
- [ ] Jupyter notebook examples
- [ ] Autoencoder comparison

## License

MIT
