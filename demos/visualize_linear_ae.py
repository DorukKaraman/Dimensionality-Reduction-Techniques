import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '..')

from src.autoencoder import LinearAutoencoder
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
    
    # Fit Linear Autoencoder
    n_components = 10
    ae = LinearAutoencoder(
        n_components=n_components,
        learning_rate=0.001,
        n_iterations=2000,
        random_state=42
    )
    ae.fit(X, verbose=True)
    
    print(f"\nLinear Autoencoder with {n_components} latent dimensions:")
    print(f"  Final loss: {ae.loss_history_[-1]:.6f}")
    
    # Reconstruct
    X_reconstructed = ae.reconstruct(X)
    images_reconstructed = matrix_to_images(X_reconstructed, image_shape)
    
    # Calculate average metrics
    psnr_vals = [psnr(images[i], images_reconstructed[i], max_val=1.0) for i in range(len(images))]
    print(f"  Average PSNR: {np.mean(psnr_vals):.2f} dB")
    
    # --- Visualization 1: Reconstructions ---
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle(f"Linear Autoencoder Reconstruction ({n_components} latent dims)", 
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
    plt.savefig("../images/linear_ae_reconstruction.png", dpi=150)
    print("\nSaved: ../images/linear_ae_reconstruction.png")
    plt.show()
    
    # --- Visualization 2: Learned Encoder Weights ---
    # Each column of W_encoder_ is a learned feature (like eigenfaces)
    fig2, axes2 = plt.subplots(1, 5, figsize=(12, 2.5))
    fig2.suptitle("Linear Autoencoder - Learned Features (Encoder Weights)", fontsize=12, fontweight='bold')
    
    for i in range(5):
        # Get i-th column of encoder weights
        feature = ae.W_encoder_[:, i].reshape(image_shape)
        axes2[i].imshow(feature, cmap='RdBu_r')
        axes2[i].set_title(f"Feature {i+1}", fontsize=9)
        axes2[i].axis('off')
    
    plt.tight_layout()
    plt.savefig("../images/linear_ae_features.png", dpi=150)
    print("Saved: ../images/linear_ae_features.png")
    plt.show()


if __name__ == "__main__":
    main()
