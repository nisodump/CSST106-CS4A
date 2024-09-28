"""Image Alignment Using Homography

- Use the matched keypoints from SIFT (or any other method) to compute a **homography matrix**.
- Use this matrix to warp one image onto the other.
- Display and save the aligned and warped images.
"""

# Import the necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the two images
image1 = cv2.imread("/content/image_1.jpg")
image2 = cv2.imread("/content/image_2.jpg")

# Convert to grayscale
gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and descriptors for both images
keypoints1, descriptors1 = sift.detectAndCompute(gray_image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray_image2, None)

# Set FLANN-based matcher parameters
index_params = dict(algorithm=1, trees=5)  # Algorithm 1 is for KD-tree
search_params = dict(checks=50)  # Higher number of checks for better precision

# Initialize FLANN matcher
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Match descriptors using K-Nearest Neighbors
matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# Apply Lowe's ratio test to filter good matches
good_matches = []
for m, n in matches:
  if m.distance < 0.75 * n.distance:
    good_matches.append(m)

# Extract location of good matches
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Compute the homography matrix using RANSAC
homography_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Warp the first image (image1) to align with the second image (image2) using the homography matrix
height, width = image2.shape[:2]
warped_image = cv2.warpPerspective(image1, homography_matrix, (width, height))

# Display the aligned image
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
plt.title("Warped Image")
plt.axis('off')
plt.show()

# Save the warped image
cv2.imwrite("warped_image.png", warped_image)

# Display both images (original and aligned) side-by-side for comparison
combined_image = np.hstack((image2, warped_image))
plt.figure(figsize=(14, 7))
plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
plt.title("Original Image (Left) and Warped Image (Right)")
plt.axis('off')
plt.show()

# Save the side-by-side comparison
cv2.imwrite("comparison_image.png", combined_image)
