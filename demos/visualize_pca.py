import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '..')

from src.svd import PCA
from src.metrics import psnr, relative_error
from data.synthetic import create_synthetic_faces, images_to_matrix, matrix_to_images


def main():
    # Create dataset
    print("Creating synthetic face dataset...")
    images = create_synthetic_faces(n_samples=50, size=32)
    image_shape = images[0].shape
    
    # Flatten to matrix
    X = images_to_matrix(images)
    print(f"Dataset shape: {X.shape} (samples x pixels)")
    
    # Fit PCA
    n_components = 10
    pca = PCA(n_components=n_components)
    pca.fit(X)
    
    print(f"\nPCA with {n_components} components:")
    print(f"  Variance explained: {sum(pca.explained_variance_ratio_):.1%}")
    
    # Transform and reconstruct
    X_transformed = pca.transform(X)
    X_reconstructed = pca.inverse_transform(X_transformed)
    images_reconstructed = matrix_to_images(X_reconstructed, image_shape)
    
    # --- Visualization ---
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle(f"PCA Reconstruction ({n_components} components, {sum(pca.explained_variance_ratio_):.0%} variance)", 
                 fontsize=12, fontweight='bold')
    
    # Show 5 samples: original (top row) vs reconstructed (bottom row)
    sample_indices = [0, 10, 20, 30, 40]
    
    for i, idx in enumerate(sample_indices):
        # Original
        axes[0, i].imshow(images[idx], cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f"Sample {idx}", fontsize=9)
        axes[0, i].axis('off')
        
        # Reconstructed
        axes[1, i].imshow(images_reconstructed[idx], cmap='gray', vmin=0, vmax=1)
        p = psnr(images[idx], images_reconstructed[idx], max_val=1.0)
        err = relative_error(images[idx], images_reconstructed[idx])
        axes[1, i].set_title(f"PSNR: {p:.1f} dB | Err: {err:.2f}", fontsize=8)
        axes[1, i].axis('off')
    
    # Row labels
    axes[0, 0].set_ylabel("Original", fontsize=10, fontweight='bold')
    axes[1, 0].set_ylabel("Reconstructed", fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("../images/pca_reconstruction.png", dpi=150)
    print("\nSaved: ../images/pca_reconstruction.png")
    plt.show()
    
    # Second figure: Principal Components
    fig2, axes2 = plt.subplots(1, 5, figsize=(12, 2.5))
    fig2.suptitle("Top 5 Principal Components (Eigenfaces)", fontsize=12, fontweight='bold')
    
    for i in range(5):
        component = pca.components_[i].reshape(image_shape)
        axes2[i].imshow(component, cmap='RdBu_r')
        axes2[i].set_title(f"PC {i+1}: {pca.explained_variance_ratio_[i]:.1%}", fontsize=9)
        axes2[i].axis('off')
    
    plt.tight_layout()
    plt.savefig("../images/pca_components.png", dpi=150)
    print("Saved: ../images/pca_components.png")
    plt.show()


if __name__ == "__main__":
    main()
