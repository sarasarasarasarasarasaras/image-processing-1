#CARTOON EFFECT IMAGE PROCESSING  README

## Introduction
This Python script performs cartoon effect using various techniques, including image smoothing, edge detection, and image quantization to an input image. The code utilizes popular computer vision libraries such as OpenCV and NumPy.

## Steps 

### 1. Image Smoothing
   - The input image is smoothed using a Gaussian blur.

### 2. Edge Detection (DoG)
   - The script uses the Difference of Gaussians (DoG) method for edge detection.

### 3. Image Quantization
   - The L channel in the LAB color space is quantized.

### 4. Combine Edge and Quantized Image
   - The quantized image and the inverse of the edge image are combined using a bitwise AND operation.

## Usage
1. Install the required libraries:
   ```bash
   pip install opencv-python numpy Pillow scipy



Update the script with your input image path:

If you want to run the code and see how it gives the output you should change the {folder} part with data and {file} with an image name that i provided in the data folder in this line "input_image = cv2.imread(r"../report/{folder}/{file}.jpg")"

