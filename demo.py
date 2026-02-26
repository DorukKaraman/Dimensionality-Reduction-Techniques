import numpy as np
import matplotlib.pyplot as plt

from svd import compress_image


def create_sample_image(size=512):
    # Checkerboard with shapes
    image = np.zeros((size, size))
    
    # Checkerboard pattern
    block = size // 8
    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 0:
                image[i*block:(i+1)*block, j*block:(j+1)*block] = 255
    
    # Circle
    y, x = np.ogrid[:size, :size]
    center = size // 2
    mask = (x - center)**2 + (y - center)**2 < (size // 4)**2
    image[mask] = 128
    
    # Diagonal lines
    for i in range(0, size, 16):
        np.fill_diagonal(image[i:], 200)
    
    return image


def visualize_compression(image, ranks):
    # Original and compressed versions side by side
    fig, axes = plt.subplots(1, len(ranks) + 1, figsize=(15, 3))
    
    axes[0].imshow(image, cmap='gray', interpolation='nearest')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    for i, k in enumerate(ranks):
        compressed = compress_image(image, k)
        axes[i + 1].imshow(compressed, cmap='gray', interpolation='nearest')
        axes[i + 1].set_title(f'Rank {k}')
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('compression_demo.png', dpi=150)
    plt.show()
    print("Saved to compression_demo.png")


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
