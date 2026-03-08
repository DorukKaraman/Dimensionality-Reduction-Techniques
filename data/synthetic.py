import numpy as np


def create_synthetic_faces(n_samples=100, size=32, random_state=42):
    """
    Create synthetic 'face-like' images with eyes and mouth patterns.
    
    Returns:
        Array of shape (n_samples, size, size)
    """
    images = []
    rng = np.random.default_rng(random_state)
    
    for _ in range(n_samples):
        img = np.ones((size, size)) * 0.8  # light background
        
        # Random variations for each face
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


def images_to_matrix(images):
    """Flatten images to row vectors (for PCA/linear methods)."""
    n_samples = images.shape[0]
    return images.reshape(n_samples, -1)


def matrix_to_images(matrix, image_shape):
    """Reshape row vectors back to images."""
    return matrix.reshape(-1, *image_shape)


def create_sample_image(size=512):
    """
    Checkerboard with shapes
    """
    image = np.zeros((size, size))
    
    # Checkerboard pattern
    block = size // 8
    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 0:
                image[i*block:(i+1)*block, j*block:(j+1)*block] = 255
    
    # Circle in center
    y, x = np.ogrid[:size, :size]
    center = size // 2
    mask = (x - center)**2 + (y - center)**2 < (size // 4)**2
    image[mask] = 128
    
    # Diagonal lines
    for i in range(0, size, 16):
        np.fill_diagonal(image[i:], 200)
    
    return image
