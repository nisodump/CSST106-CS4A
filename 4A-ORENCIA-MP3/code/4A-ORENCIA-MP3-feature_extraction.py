"""Extract Keypoints and Descriptors Using SIFT, SURF, and ORB

- Apply the SIFT algorithm to detect keypoints and compute descriptors for both images.
- Apply the SURF algorithm to do the same.
- Finally, apply ORB to extract keypoints and descriptors
"""

"""
SIFT (Scale-Invariant Feature Transform) Feature Extraction
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

# Draw keypoints on both images
image1_with_keypoints = cv2.drawKeypoints(image1, keypoints1, None)
image2_with_keypoints = cv2.drawKeypoints(image2, keypoints2, None)

# Resize the images to the same size (e.g., 500x500 pixels)
image1_resized = cv2.resize(image1_with_keypoints, (500, 500))
image2_resized = cv2.resize(image2_with_keypoints, (500, 500))

# Combine both images side-by-side
combined_image = np.hstack((image1_resized, image2_resized))

# Enlarge the display by setting a larger figure size
plt.figure(figsize=(12, 6))
plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
plt.title("SIFT Keypoints (Side-by-Side)")
plt.axis('off')
plt.show()

# Print the number of keypoints detected in each image
print(f"Number of keypoints detected in image 1: {len(keypoints1)}")
print(f"Number of keypoints detected in image 2: {len(keypoints2)}")

"""
SURF (Speeded Up Robust Features) Feature Extraction
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

# Initialize SURF detector (adjust the Hessian threshold as needed)
surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400)

# Detect keypoints and descriptors for both images
keypoints1, descriptors1 = surf.detectAndCompute(gray_image1, None)
keypoints2, descriptors2 = surf.detectAndCompute(gray_image2, None)

# Draw keypoints on both images
image1_with_keypoints = cv2.drawKeypoints(image1, keypoints1, None)
image2_with_keypoints = cv2.drawKeypoints(image2, keypoints2, None)

# Resize the images to the same size (e.g., 500x500 pixels)
image1_resized = cv2.resize(image1_with_keypoints, (500, 500))
image2_resized = cv2.resize(image2_with_keypoints, (500, 500))

# Combine both images side-by-side
combined_image = np.hstack((image1_resized, image2_resized))

# Enlarge the display by setting a larger figure size
plt.figure(figsize=(12, 6))
plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
plt.title("SURF Keypoints Comparison (Side-by-Side)")
plt.axis('off')
plt.show()

# Print the number of keypoints detected in each image
print(f"Number of keypoints detected in image 1: {len(keypoints1)}")
print(f"Number of keypoints detected in image 2: {len(keypoints2)}")

"""
ORB (Oriented FAST and Rotated BRIEF) Feature Extraction
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

# Initialize ORB detector (set the number of keypoints to detect)
orb = cv2.ORB_create(nfeatures=500)

# Detect keypoints and descriptors for both images
keypoints1, descriptors1 = orb.detectAndCompute(gray_image1, None)
keypoints2, descriptors2 = orb.detectAndCompute(gray_image2, None)

# Draw keypoints on both images
image1_with_keypoints = cv2.drawKeypoints(image1, keypoints1, None, color=(0, 255, 0))
image2_with_keypoints = cv2.drawKeypoints(image2, keypoints2, None, color=(0, 255, 0))

# Resize the images to the same size (e.g., 500x500 pixels)
image1_resized = cv2.resize(image1_with_keypoints, (500, 500))
image2_resized = cv2.resize(image2_with_keypoints, (500, 500))

# Combine both images side-by-side
combined_image = np.hstack((image1_resized, image2_resized))

# Enlarge the display by setting a larger figure size
plt.figure(figsize=(12, 6))
plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
plt.title("ORB Keypoints Comparison (Side-by-Side)")
plt.axis('off')
plt.show()

# Print the number of keypoints detected in each image
print(f"Number of keypoints detected in image 1: {len(keypoints1)}")
print(f"Number of keypoints detected in image 2: {len(keypoints2)}")
