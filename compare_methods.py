import numpy as np
import matplotlib.pyplot as plt
from svd import TruncatedSVD, PCA
from autoencoder import LinearAutoencoder, ConvAutoencoder
from metrics import psnr, relative_error


def create_synthetic_dataset(n_samples=100, size=32):
    # Create synthetic face-like images
    images = []
    rng = np.random.default_rng(42)
    
    for _ in range(n_samples):
        img = np.ones((size, size)) * 0.8
        
        eye_y = size // 3 + rng.integers(-2, 3)
        eye_size = 2 + rng.integers(0, 2)
        mouth_y = 2 * size // 3 + rng.integers(-2, 3)
        
        # Eyes
        left_x = size // 3 + rng.integers(-2, 3)
        right_x = 2 * size // 3 + rng.integers(-2, 3)
        img[eye_y-eye_size:eye_y+eye_size, left_x-eye_size:left_x+eye_size] = 0.2
        img[eye_y-eye_size:eye_y+eye_size, right_x-eye_size:right_x+eye_size] = 0.2
        
        # Mouth
        mouth_width = size // 4 + rng.integers(-2, 3)
        img[mouth_y:mouth_y+2, size//2-mouth_width:size//2+mouth_width] = 0.3
        
        img += rng.normal(0, 0.05, (size, size))
        images.append(np.clip(img, 0, 1))
    
    return np.array(images)


def main():
    print("Creating synthetic dataset...")
    images = create_synthetic_dataset(n_samples=100, size=32)
    image_shape = (32, 32)
    
    # Flatten to matrix (each row = one image)
    X = images.reshape(len(images), -1)
    print(f"Dataset: {X.shape[0]} images, {X.shape[1]} pixels each")
    
    n_components = 10
    print(f"\nCompressing to {n_components} components...")
    
    # --- Method 1: PCA ---
    print("\n[PCA]")
    pca = PCA(n_components=n_components)
    pca.fit(X)
    X_pca = pca.inverse_transform(pca.transform(X))
    print(f"  Variance explained: {pca.explained_variance_ratio_.sum():.1%}")
    
    # --- Method 2: Linear Autoencoder ---
    print("\n[Linear Autoencoder]")
    linear_ae = LinearAutoencoder(n_components=n_components, learning_rate=0.1, 
                                   n_iterations=500, random_state=42)
    linear_ae.fit(X, verbose=False)
    X_linear = linear_ae.reconstruct(X)
    print(f"  Final loss: {linear_ae.loss_history_[-1]:.6f}")
    
    # --- Method 3: Convolutional Autoencoder ---
    print("\n[Conv Autoencoder]")
    conv_ae = ConvAutoencoder(n_filters=16, latent_channels=4, learning_rate=0.001,
                               n_epochs=100, batch_size=16, random_state=42)
    conv_ae.fit(images, verbose=False)
    X_conv = conv_ae.reconstruct(images)
    print(f"  Final loss: {conv_ae.loss_history_[-1]:.6f}")
    
    # --- Visualization ---
    fig, axes = plt.subplots(4, 6, figsize=(14, 9))
    fig.suptitle(f"PCA vs Linear AE vs Conv AE ({n_components} components / latent dims)", 
                 fontsize=14, fontweight='bold')
    
    sample_indices = [0, 20, 40, 60, 80]
    
    # Row labels
    row_labels = ["Original", "PCA", "Linear AE", "Conv AE"]
    for row, label in enumerate(row_labels):
        axes[row, 0].annotate(label, xy=(-0.3, 0.5), xycoords='axes fraction',
                              fontsize=11, fontweight='bold', ha='right', va='center')
    
    for i, idx in enumerate(sample_indices):
        orig = images[idx]
        pca_img = X_pca[idx].reshape(image_shape)
        linear_img = np.clip(X_linear[idx].reshape(image_shape), 0, 1)
        conv_img = np.clip(X_conv[idx], 0, 1)
        
        # Original
        axes[0, i].imshow(orig, cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f"#{idx}", fontsize=9)
        axes[0, i].axis('off')
        
        # PCA
        axes[1, i].imshow(pca_img, cmap='gray', vmin=0, vmax=1)
        p = psnr(orig, pca_img, max_val=1.0)
        axes[1, i].set_title(f"PSNR: {p:.1f}", fontsize=8)
        axes[1, i].axis('off')
        
        # Linear Autoencoder
        axes[2, i].imshow(linear_img, cmap='gray', vmin=0, vmax=1)
        p = psnr(orig, linear_img, max_val=1.0)
        axes[2, i].set_title(f"PSNR: {p:.1f}", fontsize=8)
        axes[2, i].axis('off')
        
        # Conv Autoencoder
        axes[3, i].imshow(conv_img, cmap='gray', vmin=0, vmax=1)
        p = psnr(orig, conv_img, max_val=1.0)
        axes[3, i].set_title(f"PSNR: {p:.1f}", fontsize=8)
        axes[3, i].axis('off')
    
    # Info in last column
    axes[0, 5].axis('off')
    
    # PCA info (row 1, next to PCA images)
    axes[1, 5].axis('off')
    axes[1, 5].text(0.5, 0.5, f"PCA Var:\n{pca.explained_variance_ratio_.sum():.1%}", 
                    ha='center', va='center', fontsize=10)
    
    # Linear AE loss curve (row 2, next to Linear AE images)
    axes[2, 5].plot(linear_ae.loss_history_, 'b-', linewidth=1)
    axes[2, 5].set_title("Linear AE Loss", fontsize=8)
    axes[2, 5].set_xlabel("Iter", fontsize=7)
    axes[2, 5].set_ylabel("Loss", fontsize=7)
    axes[2, 5].tick_params(labelsize=6)
    
    # Conv AE loss curve (row 3, next to Conv AE images)
    axes[3, 5].plot(conv_ae.loss_history_, 'r-', linewidth=1)
    axes[3, 5].set_title("Conv AE Loss", fontsize=8)
    axes[3, 5].set_xlabel("Epoch", fontsize=7)
    axes[3, 5].set_ylabel("Loss", fontsize=7)
    axes[3, 5].tick_params(labelsize=6)
    
    plt.tight_layout()
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.12)  # Make room for row labels
    plt.savefig("comparison_all_methods.png", dpi=150)
    print("\nSaved: comparison_all_methods.png")
    plt.show()
    
    # Print summary
    print("\n" + "="*50)
    print("Summary (average over all images)")
    print("="*50)
    
    pca_psnr = np.mean([psnr(images[i], X_pca[i].reshape(image_shape), max_val=1.0) 
                        for i in range(len(images))])
    linear_psnr = np.mean([psnr(images[i], np.clip(X_linear[i].reshape(image_shape), 0, 1), max_val=1.0) 
                           for i in range(len(images))])
    conv_psnr = np.mean([psnr(images[i], np.clip(X_conv[i], 0, 1), max_val=1.0) 
                         for i in range(len(images))])
    
    print(f"PCA:            {pca_psnr:.2f} dB average PSNR")
    print(f"Linear AE:      {linear_psnr:.2f} dB average PSNR")
    print(f"Conv AE:        {conv_psnr:.2f} dB average PSNR")
    print("\nNote: Conv AE can capture spatial patterns better than linear methods.")


if __name__ == "__main__":
    main()
