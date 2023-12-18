'''
Martin Poovathingal
AER 850 - Fall 2023
Project 3 - 
Dec 17, 2023
'''
import cv2
import numpy as np

# Load the motherboard image
image = cv2.imread('motherboard_image.JPEG')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding to segment the image
_, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

# Apply morphological operations to enhance features
kernel = np.ones((5, 5), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Used to tweak contrast adjustment, decided not to apply
alpha = 1
beta = 1
contrast_adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# Apply edge detection using Canny
edges = cv2.Canny(thresh, 30, 100)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours based on area or size
filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > 1000]

# Create a mask
mask = np.zeros_like(gray)
cv2.drawContours(mask, filtered_contours, -1, (255), thickness=cv2.FILLED)

# Bitwise AND to extract the PCB from the background
result = cv2.bitwise_and(contrast_adjusted, contrast_adjusted, mask=mask)

# Resize the image for display
height, width = result.shape[:2]
max_display_height = 1080  # Set to my monitor height
if height > max_display_height:
    scale_factor = max_display_height / height
    resized_result = cv2.resize(result, (int(width * scale_factor), max_display_height))
    resized_edges = cv2.resize(edges, (int(width * scale_factor), max_display_height))
    resized_mask = cv2.resize(mask, (int(width * scale_factor), max_display_height))

# Save the result
cv2.imwrite('Edge_Detected_Motherboard.jpg', resized_edges)
cv2.imwrite('Masked_Motherboard.jpg', resized_mask)
cv2.imwrite('Final_Extracted_Motherboard.jpg', resized_result)
# cv2.waitKey(0) 
# cv2.destroyAllWindows() 



