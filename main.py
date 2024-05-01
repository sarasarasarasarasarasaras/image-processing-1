import cv2
import numpy as np
from PIL import ImageFilter
from scipy.ndimage import gaussian_filter
import scipy.ndimage as ndimage


# Step 1: Image Smoothing
def smooth_image(input_image, sigma):
    return cv2.GaussianBlur(input_image, (5, 5), 5)



# Step 2: Edge Detection (DoG)
def edge_detection(input_image, ksize=(5, 5), ksigma=1.0, k=2.0, threshold=100):
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    g1 = cv2.GaussianBlur(gray_image, ksize, ksigma)
    g2 = cv2.GaussianBlur(gray_image, ksize, ksigma * k)
    dog = g1 - g2
    edge_image = cv2.threshold(dog, threshold, 255, cv2.THRESH_BINARY)[1]
    return edge_image


# Step 3: Image Quantization
def quantize_image(input_image, channels_to_quantize=['L']):
    lab_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2Lab)
    if 'L' in channels_to_quantize:
        lab_image[:, :, 0] = (lab_image[:, :, 0] // 13) * 10 # Quantize the L channel
    quantized_image = cv2.cvtColor(lab_image, cv2.COLOR_Lab2BGR)
    return quantized_image

# Step 4: Combine Edge and Quantized Image
def combine_images(quantized_image, edge_image):
    inverse_edges = 255 - edge_image
    combined_image = cv2.bitwise_and(quantized_image, quantized_image, mask=inverse_edges)
    return combined_image

# Load the input image
input_image = cv2.imread(r"..\report\{folder}\{file}.jpg")



# Apply image abstraction step
smoothed = smooth_image(input_image, sigma=10)
edges = edge_detection(smoothed, ksize=(1, 5), ksigma=2.4, k=2.5, threshold=10)
quantized = quantize_image(smoothed, channels_to_quantize=['L'])
result = combine_images(quantized, edges)

# Save the output image


output_path = (r"..\{folder}\output_edges.jpg")
cv2.imwrite(output_path, edges)

output_path = (r"..\{folder}\output_smoothes.jpg")
cv2.imwrite(output_path, smoothed)

output_path = (r"..\{folder}\output_quantized.jpg")
cv2.imwrite(output_path, quantized)


output_path = (r"..\{folder}\output_image.jpg")
cv2.imwrite(output_path, result)





