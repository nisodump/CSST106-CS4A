"""Feature Matching with Brute-Force and FLANN

- Match the descriptors between the two images using **Brute-Force Matcher**.
- Repeat the process using the **FLANN Matcher**.
- For each matching method, display the matches with lines connecting corresponding keypoints between the two images.
"""

"""
Feature Matching with Brute Force Matcher using ORB
"""

# Import the necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

def resize_with_aspect_ratio(image, target_width):
    """Resize an image while maintaining its aspect ratio."""
    h, w = image.shape[:2]
    aspect_ratio = w / h
    new_width = target_width
    new_height = int(new_width / aspect_ratio)
    return cv2.resize(image, (new_width, new_height))

# Load the two images
image1 = cv2.imread("/content/image_1.jpg")
image2 = cv2.imread("/content/image_2.jpg")

# Resize images to have the same width (e.g., 500 pixels) while preserving aspect ratio
target_width = 500
image1_resized = resize_with_aspect_ratio(image1, target_width)
image2_resized = resize_with_aspect_ratio(image2, target_width)

# Convert to grayscale
gray_image1 = cv2.cvtColor(image1_resized, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(image2_resized, cv2.COLOR_BGR2GRAY)

# Initialize ORB detector (set the number of keypoints to detect)
orb = cv2.ORB_create(nfeatures=500)

# Detect keypoints and descriptors for both images
keypoints1, descriptors1 = orb.detectAndCompute(gray_image1, None)
keypoints2, descriptors2 = orb.detectAndCompute(gray_image2, None)

# Initialize Brute-Force Matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors
matches = bf.match(descriptors1, descriptors2)

# Sort matches based on their distances (best matches first)
matches = sorted(matches, key=lambda x: x.distance)

# Draw matches with lines connecting corresponding keypoints between images
matched_image = cv2.drawMatches(image1_resized, keypoints1, image2_resized, keypoints2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the image with matches
plt.figure(figsize=(14, 7))
plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
plt.title("Feature Matching with Brute Force Matcher using ORB")
plt.axis('off')
plt.show()

# Print the number of matches
print(f"Number of matches: {len(matches)}")

"""
Feature Matching with FLANN Matcher using ORB
"""

# Import the necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

def resize_with_aspect_ratio(image, target_width):
    """Resize an image while maintaining its aspect ratio."""
    h, w = image.shape[:2]
    aspect_ratio = w / h
    new_width = target_width
    new_height = int(new_width / aspect_ratio)
    return cv2.resize(image, (new_width, new_height))

# Load the two images
image1 = cv2.imread("/content/image_1.jpg")
image2 = cv2.imread("/content/image_2.jpg")

# Resize images to have the same width (e.g., 500 pixels) while preserving aspect ratio
target_width = 500
image1_resized = resize_with_aspect_ratio(image1, target_width)
image2_resized = resize_with_aspect_ratio(image2, target_width)

# Convert to grayscale
gray_image1 = cv2.cvtColor(image1_resized, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(image2_resized, cv2.COLOR_BGR2GRAY)

# Initialize ORB detector (set the number of keypoints to detect)
orb = cv2.ORB_create(nfeatures=500)

# Detect keypoints and descriptors for both images
keypoints1, descriptors1 = orb.detectAndCompute(gray_image1, None)
keypoints2, descriptors2 = orb.detectAndCompute(gray_image2, None)

# Set FLANN-based matcher parameters for binary descriptors
index_params = dict(algorithm=6,
                    table_number=12,
                    key_size=20,
                    multi_probe_level=2)

search_params = dict(checks=50)  # Higher number of checks increases precision, but slower

# Initialize FLANN matcher
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Match descriptors using K-Nearest Neighbors
matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# Apply Lowe's ratio test to filter good matches
good_matches = []
for match in matches:
  # Ensure there are at least two matches for the current keypoint
  if len(match) == 2:
    m, n = match
    if m.distance < 0.75 * n.distance:
      good_matches.append(m)

# Draw matches with lines connecting corresponding keypoints between images
matched_image = cv2.drawMatches(image1_resized, keypoints1, image2_resized, keypoints2, good_matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the image with matches
plt.figure(figsize=(14, 7))
plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
plt.title("Feature Matching with FLANN Matcher using ORB")
plt.axis('off')
plt.show()

# Print the number of good matches
print(f"Number of good matches: {len(good_matches)}")
