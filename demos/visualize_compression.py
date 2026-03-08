import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '..')

from src.svd import TruncatedSVD
from src.metrics import relative_error, psnr
from data.synthetic import create_sample_image


def visualize_compression(image, ranks):
    # Original and compressed versions side by side
    fig, axes = plt.subplots(1, len(ranks) + 1, figsize=(15, 4))
    
    axes[0].imshow(image, cmap='gray', interpolation='nearest')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    for i, k in enumerate(ranks):
        svd = TruncatedSVD(n_components=k)
        svd.fit(image)
        compressed = np.clip(svd.reconstruct(), 0, 255)
        
        # Compute metrics
        rel_err = relative_error(image, compressed) * 100
        psnr_val = psnr(image, compressed)
        var_explained = svd.explained_variance_ratio_.sum() * 100
        
        axes[i + 1].imshow(compressed, cmap='gray', interpolation='nearest')
        axes[i + 1].set_title(f'Rank {k}\nPSNR: {psnr_val:.1f} dB\nError: {rel_err:.1f}%')
        axes[i + 1].axis('off')
        
        print(f"Rank {k:3d}: PSNR={psnr_val:5.1f} dB, RelErr={rel_err:5.1f}%, VarExpl={var_explained:5.1f}%")
    
    plt.tight_layout()
    plt.savefig('../images/svd_compression_demo.png', dpi=150)
    plt.show()
    print("\nSaved to ../images/svd_compression_demo.png")


def main():
    print("SVD Image Compression Demo")
    print("-" * 30)
    
    image = create_sample_image(512)
    print(f"Image shape: {image.shape}")
    
    ranks = [5, 20, 50, 100]
    print(f"Testing ranks: {ranks}")
    
    visualize_compression(image, ranks)


if __name__ == "__main__":
    main()
