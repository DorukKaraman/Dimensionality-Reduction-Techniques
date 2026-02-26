# Dimensionality Reduction Techniques for Images

SVD-based image compression demonstrating the Eckart-Young-Mirsky theorem.

## Quick Start

```bash
pip install -r requirements.txt
python test_svd.py
```

## Demo

```bash
python demo.py
```

Generates a visualization comparing the original image with compressed versions at different ranks (5, 20, 50, 100).

## Usage

```python
from svd import compress_image
import numpy as np

# Load your grayscale image as numpy array
image = ...  # shape (H, W)

# Compress with rank-50 approximation
compressed = compress_image(image, k=50)
```

## Roadmap

- [ ] Add PCA for face datasets (eigenfaces)
- [ ] Visualization utilities
- [ ] Reconstruction error metrics
- [ ] Jupyter notebook examples
- [ ] Autoencoder comparison
