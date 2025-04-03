import cv2
import numpy as np
import random
from matplotlib import pyplot as plt

# Use the absolute path to the image
image_path = 'C:/Users/user/Desktop/cv task 1/gray.jpg'

image_cv = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply Canny edge detection
edges = cv2.Canny(image_cv, 100, 200)

# Plot the original image and the result of the edge detection
plt.figure(figsize=(10, 5))

# Original grayscale image
plt.subplot(1, 2, 1)
plt.imshow(image_cv, cmap='gray')
plt.title('Original Grayscale Image')
plt.axis('off')

# Canny edge detection result
plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title('Canny Edge Detection')
plt.axis('off')

# Display the results
plt.show()

# Function to add Salt and Pepper noise
def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    noisy_image = np.copy(image)
    total_pixels = image.size
    
    # Salt noise
    num_salt = np.ceil(salt_prob * total_pixels)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 255

    # Pepper noise
    num_pepper = np.ceil(pepper_prob * total_pixels)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 0

    return noisy_image

# Add Salt and Pepper noise to the image
salt_prob = 0.2  # Probability of salt noise
pepper_prob = 0.2  # Probability of pepper noise
noisy_image = add_salt_and_pepper_noise(image_cv, salt_prob, pepper_prob)

# Apply average filtering with different filter sizes
filtered_image_3x3 = cv2.blur(noisy_image, (3, 3))
filtered_image_5x5 = cv2.blur(noisy_image, (5, 5))
filtered_image_9x9 = cv2.blur(noisy_image, (9, 9))

# Plot 1: Original vs Noisy image
plt.figure(figsize=(10, 5))

# Original grayscale image
plt.subplot(1, 2, 1)
plt.imshow(image_cv, cmap='gray')
plt.title('Original Grayscale Image')
plt.axis('off')

# Noisy image with Salt and Pepper noise
plt.subplot(1, 2, 2)
plt.imshow(noisy_image, cmap='gray')
plt.title('Image with Salt and Pepper Noise')
plt.axis('off')

# Show the first plot
plt.show()

# Plot 2: Original vs Noisy vs Filtered images
plt.figure(figsize=(20, 10))

# Original grayscale image
plt.subplot(2, 3, 1)
plt.imshow(image_cv, cmap='gray')
plt.title('Original Grayscale Image')
plt.axis('off')

# Noisy image with Salt and Pepper noise
plt.subplot(2, 3, 2)
plt.imshow(noisy_image, cmap='gray')
plt.title('Image with Salt and Pepper Noise')
plt.axis('off')

# Filtered image with 3x3 average filtering
plt.subplot(2, 3, 3)
plt.imshow(filtered_image_3x3, cmap='gray')
plt.title('Image after 3x3 Average Filtering')
plt.axis('off')

# Filtered image with 5x5 average filtering
plt.subplot(2, 3, 4)
plt.imshow(filtered_image_5x5, cmap='gray')
plt.title('Image after 5x5 Average Filtering')
plt.axis('off')

# Filtered image with 9x9 average filtering
plt.subplot(2, 3, 5)
plt.imshow(filtered_image_9x9, cmap='gray')
plt.title('Image after 9x9 Average Filtering')
plt.axis('off')

# Show the second plot
plt.show()



# Function to add Gaussian noise
def add_gaussian_noise(image, mean=0, var=0.01):
    row, col = image.shape
    sigma = var**0.5  # Standard deviation
    gaussian = np.random.normal(mean, sigma, (row, col))  # Gaussian noise
    noisy_image = image + gaussian * 255  # Scale to image pixel range
    noisy_image = np.clip(noisy_image, 0, 255)  # Clip pixel values to stay within valid range
    return noisy_image.astype(np.uint8)

# Add Gaussian noise to the image
mean = 0.3  # Mean of the Gaussian distribution
var = 0.1   # Variance of the Gaussian noise
noisy_image = add_gaussian_noise(image_cv, mean, var)

# Apply average filtering with different filter sizes
filtered_image_3x3 = cv2.blur(noisy_image, (3, 3))
filtered_image_5x5 = cv2.blur(noisy_image, (5, 5))
filtered_image_9x9 = cv2.blur(noisy_image, (9, 9))

# Plot 1: Original vs Noisy image
plt.figure(figsize=(10, 5))

# Original grayscale image
plt.subplot(1, 2, 1)
plt.imshow(image_cv, cmap='gray')
plt.title('Original Grayscale Image')
plt.axis('off')

# Noisy image with Gaussian noise
plt.subplot(1, 2, 2)
plt.imshow(noisy_image, cmap='gray')
plt.title('Image with Gaussian Noise')
plt.axis('off')

# Show the first plot (Original vs Noisy)
plt.show()

# Plot 2: Original vs Noisy vs Filtered images
plt.figure(figsize=(20, 10))

# Original grayscale image
plt.subplot(2, 3, 1)
plt.imshow(image_cv, cmap='gray')
plt.title('Original Grayscale Image')
plt.axis('off')

# Noisy image with Gaussian noise
plt.subplot(2, 3, 2)
plt.imshow(noisy_image, cmap='gray')
plt.title('Image with Gaussian Noise')
plt.axis('off')

# Filtered image with 3x3 average filtering
plt.subplot(2, 3, 3)
plt.imshow(filtered_image_3x3, cmap='gray')
plt.title('Image after 3x3 Average Filtering')
plt.axis('off')

# Filtered image with 5x5 average filtering
plt.subplot(2, 3, 4)
plt.imshow(filtered_image_5x5, cmap='gray')
plt.title('Image after 5x5 Average Filtering')
plt.axis('off')

# Filtered image with 9x9 average filtering
plt.subplot(2, 3, 5)
plt.imshow(filtered_image_9x9, cmap='gray')
plt.title('Image after 9x9 Average Filtering')
plt.axis('off')

# Show the second plot (Original vs Noisy vs Filtered)
plt.show()



# Function to add Poisson noise
def add_poisson_noise(image):
    noisy_image = np.copy(image).astype(np.float32)
    
    # Normalize the image to [0, 1] range
    noisy_image = noisy_image / 255.0
    
    # Apply Poisson noise and multiply the noise effect to make it more visible
    noisy_image = np.random.poisson(noisy_image * 255) / 255.0 * 2  # Multiply by a factor to increase noise visibility
    
    # Scale back to the [0, 255] range and convert to uint8
    noisy_image = np.clip(noisy_image * 255, 0, 255).astype(np.uint8)
    
    return noisy_image


# Add Poisson noise to the image
noisy_image = add_poisson_noise(image_cv)

# Apply average filtering with different filter sizes
filtered_image_3x3 = cv2.blur(noisy_image, (3, 3))
filtered_image_5x5 = cv2.blur(noisy_image, (5, 5))
filtered_image_9x9 = cv2.blur(noisy_image, (9, 9))

# Plot 1: Original vs Noisy image
plt.figure(figsize=(10, 5))

# Original grayscale image
plt.subplot(1, 2, 1)
plt.imshow(image_cv, cmap='gray')
plt.title('Original Grayscale Image')
plt.axis('off')

# Noisy image with Poisson noise
plt.subplot(1, 2, 2)
plt.imshow(noisy_image, cmap='gray')
plt.title('Image with Poisson Noise')
plt.axis('off')

# Show the first plot (Original vs Noisy)
plt.show()

# Plot 2: Original vs Noisy vs Filtered images
plt.figure(figsize=(20, 10))

# Original grayscale image
plt.subplot(2, 3, 1)
plt.imshow(image_cv, cmap='gray')
plt.title('Original Grayscale Image')
plt.axis('off')

# Noisy image with Poisson noise
plt.subplot(2, 3, 2)
plt.imshow(noisy_image, cmap='gray')
plt.title('Image with Poisson Noise')
plt.axis('off')

# Filtered image with 3x3 average filtering
plt.subplot(2, 3, 3)
plt.imshow(filtered_image_3x3, cmap='gray')
plt.title('Image after 3x3 Average Filtering')
plt.axis('off')

# Filtered image with 5x5 average filtering
plt.subplot(2, 3, 4)
plt.imshow(filtered_image_5x5, cmap='gray')
plt.title('Image after 5x5 Average Filtering')
plt.axis('off')

# Filtered image with 9x9 average filtering
plt.subplot(2, 3, 5)
plt.imshow(filtered_image_9x9, cmap='gray')
plt.title('Image after 9x9 Average Filtering')
plt.axis('off')

# Show the second plot (Original vs Noisy vs Filtered)
plt.show()



# Function to add Random noise
def add_random_noise(image, noise_factor=0.05):
    noisy_image = np.copy(image).astype(np.float32)
    
    # Generate random noise in the range [0, 255]
    random_noise = np.random.uniform(0, 255, image.shape)
    
    # Scale the random noise by a factor and add it to the image
    noisy_image = noisy_image + noise_factor * random_noise
    
    # Clip the image to stay within valid pixel range [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    return noisy_image


# Add Random noise to the image
noise_factor = 0.4  # Adjust this value to control the amount of noise
noisy_image = add_random_noise(image_cv, noise_factor)

# Apply average filtering with different filter sizes
filtered_image_3x3 = cv2.blur(noisy_image, (3, 3))
filtered_image_5x5 = cv2.blur(noisy_image, (5, 5))
filtered_image_9x9 = cv2.blur(noisy_image, (9, 9))

# Plot 1: Original vs Noisy image
plt.figure(figsize=(10, 5))

# Original grayscale image
plt.subplot(1, 2, 1)
plt.imshow(image_cv, cmap='gray')
plt.title('Original Grayscale Image')
plt.axis('off')

# Noisy image with Random noise
plt.subplot(1, 2, 2)
plt.imshow(noisy_image, cmap='gray')
plt.title('Image with Random Noise')
plt.axis('off')

# Show the first plot (Original vs Noisy)
plt.show()

# Plot 2: Original vs Noisy vs Filtered images
plt.figure(figsize=(20, 10))

# Original grayscale image
plt.subplot(2, 3, 1)
plt.imshow(image_cv, cmap='gray')
plt.title('Original Grayscale Image')
plt.axis('off')

# Noisy image with Random noise
plt.subplot(2, 3, 2)
plt.imshow(noisy_image, cmap='gray')
plt.title('Image with Random Noise')
plt.axis('off')

# Filtered image with 3x3 average filtering
plt.subplot(2, 3, 3)
plt.imshow(filtered_image_3x3, cmap='gray')
plt.title('Image after 3x3 Average Filtering')
plt.axis('off')

# Filtered image with 5x5 average filtering
plt.subplot(2, 3, 4)
plt.imshow(filtered_image_5x5, cmap='gray')
plt.title('Image after 5x5 Average Filtering')
plt.axis('off')

# Filtered image with 9x9 average filtering
plt.subplot(2, 3, 5)
plt.imshow(filtered_image_9x9, cmap='gray')
plt.title('Image after 9x9 Average Filtering')
plt.axis('off')

# Show the second plot (Original vs Noisy vs Filtered)
plt.show()