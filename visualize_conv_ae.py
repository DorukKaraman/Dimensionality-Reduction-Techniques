import numpy as np
import matplotlib.pyplot as plt
from autoencoder import ConvAutoencoder
from metrics import psnr, relative_error


def create_synthetic_faces(n_samples=100, size=32):
    """
    Create synthetic 'face-like' images with eyes and mouth patterns.
    
    Returns:
        Array of shape (n_samples, size, size)
    """
    images = []
    rng = np.random.default_rng(42)
    
    for _ in range(n_samples):
        img = np.ones((size, size)) * 0.8  # light background
        
        # Random variations
        eye_y = size // 3 + rng.integers(-2, 3)
        eye_size = 2 + rng.integers(0, 2)
        mouth_y = 2 * size // 3 + rng.integers(-2, 3)
        
        # Left eye
        left_x = size // 3 + rng.integers(-2, 3)
        img[eye_y-eye_size:eye_y+eye_size, left_x-eye_size:left_x+eye_size] = 0.2
        
        # Right eye
        right_x = 2 * size // 3 + rng.integers(-2, 3)
        img[eye_y-eye_size:eye_y+eye_size, right_x-eye_size:right_x+eye_size] = 0.2
        
        # Mouth
        mouth_width = size // 4 + rng.integers(-2, 3)
        img[mouth_y:mouth_y+2, size//2-mouth_width:size//2+mouth_width] = 0.3
        
        # Add noise
        img += rng.normal(0, 0.05, (size, size))
        img = np.clip(img, 0, 1)
        
        images.append(img)
    
    return np.array(images)


def main():
    # Create dataset
    print("Creating synthetic face dataset...")
    images = create_synthetic_faces(n_samples=100, size=32)
    print(f"Dataset shape: {images.shape} (samples x height x width)")
    
    # Fit Convolutional Autoencoder
    print("\nTraining Convolutional Autoencoder...")
    conv_ae = ConvAutoencoder(
        n_filters=16,
        latent_channels=4,
        learning_rate=0.001,
        n_epochs=100,
        batch_size=16,
        random_state=42
    )
    conv_ae.fit(images, verbose=True)
    
    print(f"\nConv Autoencoder ({conv_ae.n_filters} filters, {conv_ae.latent_channels} latent channels):")
    print(f"  Final loss: {conv_ae.loss_history_[-1]:.6f}")
    
    # Reconstruct
    images_reconstructed = conv_ae.reconstruct(images)
    
    # Calculate average metrics
    psnr_vals = [psnr(images[i], images_reconstructed[i], max_val=1.0) for i in range(len(images))]
    print(f"  Average PSNR: {np.mean(psnr_vals):.2f} dB")
    
    # --- Visualization 1: Reconstructions ---
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle(f"Conv Autoencoder Reconstruction ({conv_ae.n_filters} filters, {conv_ae.latent_channels} latent ch.)", 
                 fontsize=12, fontweight='bold')
    
    # Show 5 samples: original (top row) vs reconstructed (bottom row)
    sample_indices = [0, 20, 40, 60, 80]
    
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
    plt.savefig("conv_ae_reconstruction.png", dpi=150)
    print("\nSaved: conv_ae_reconstruction.png")
    plt.show()
    
    # --- Visualization 2: Learned Conv Filters ---
    filters = conv_ae.get_encoder_filters()  # (n_filters, kH, kW)
    n_show = min(8, filters.shape[0])
    
    fig2, axes2 = plt.subplots(1, n_show, figsize=(12, 2))
    fig2.suptitle("Conv Autoencoder - First Layer Filters", fontsize=12, fontweight='bold')
    
    for i in range(n_show):
        axes2[i].imshow(filters[i], cmap='RdBu_r')
        axes2[i].set_title(f"Filter {i+1}", fontsize=9)
        axes2[i].axis('off')
    
    plt.tight_layout()
    plt.savefig("conv_ae_filters.png", dpi=150)
    print("Saved: conv_ae_filters.png")
    plt.show()


if __name__ == "__main__":
    main()
