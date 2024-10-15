# Advanced Feature Extraction and Image Processing
## Exercise 1: Harris Detection Corner
### Task
**Harris Corner Detection** is a classic corner detection algorithm. Use the Harris Corner Detection algorithm to detect corners in an image:
* Load an image of your choice.
* Convert it to grayscale.
* Apply the Harris Corner Detection method to detect corners.
* Visualize the corners on the image and display the result.

![Harris Detection Corner](https://github.com/user-attachments/assets/8597dbc0-1158-472e-a30c-fc59c1fcbb4f)

The Harris Corner Detection algorithm bases its operation on a change in intensity as a small region of pixels shifts across the image. Corners are defined as a point with which the intensity varies in many directions so significantly that the checkerboard is highly intuitive to apply to the algorithm used here. A checkerboard image is basically an image of a grid with black and white squares alternating, thus giving a dense concentration of high contrast edges at corners where there are points of interest, that is, keen edges. 

In this task, the algorithm has detected 1365 corners which meant that it therefore manifests that many keen points of interest, that is, corners have been detected where such intense changes in gradient take place. The algorithm computes a corner response function that is the amount of two-dimensional intensity change for each pixel, such as along the edges or corners of checkerboard squares. Checkerboards are a particularly good test case since the checkerboard structure has to be uniform and well-distributed, and thus a dense grid of corners provides ample opportunity for quite precise detection and evaluation of corner-detection algorithms.  The number of the actual corners detected should be very high, as the pattern is periodic and symmetrical.
### Key Points
* HOG focuses on the structure of objects through gradients.
* Useful for human detection and general object recognition.

## Exercise 2: HOG (Histogram of Oriented Gradients) Feature Extraction
### Task
The **HOG descriptor** is widely used for object detection, especially in human detection.
* Load an image of a person or any object.
* Convert the image to grayscale.
* Apply the HOG descriptor to extract features.
* Visualize the gradient orientations on the image

![HOG (Histogram of Oriented Gradients) Feature Extraction](https://github.com/user-attachments/assets/ecd5405f-1379-4529-a4ec-b3a91f9d97eb)

As seen above, HOG Feature Extraction is applied to an image of two pedestrians who are walking. HOG Feature Extraction resulted in features at a total of 2,047,752 for all the same shape. Such a high number of features denotes a detailed encoding of the image's local gradients and edge orientations. The HOG algorithm works by breaking the image down into small cells and computing a histogram of gradient orientations within each cell. These captures shape information, for it highlights regions of high gradient change, like edges and contours—rather than colors or textures. 

Pedestrian images fit perfectly well with this approach since the human body, especially when moving, presents remarkable contours and shapes. The human form, with its limbs and outlines, produces significant gradients, which HOG captures and summarizes effectively using those gradient histograms. It is robust in detecting people under varying conditions of lighting, scale, or slight pose changes and is thus ideal for pedestrian detection. Several features represent the rich detail captured by the algorithm, which can be represented so well that a model trained on these features may differentiate pedestrians from background with high precision. Its strength is pointed at focusing on the most relevant shape and edge features because these are central aspects that carry information necessary for such tasks as detecting humans in images.

### Key Points
* HOG focuses on the structure of objects through gradients.
* Useful for human detection and general object recognition.

## Exercise 3: FAST (Features from Accelerated Segment Test) Keypoint Detection
### Task
**FAST** is another keypoint detector known for its speed.
* Load an image.
* Convert the image to grayscale.
* Apply the FAST algorithm to detect keypoints.
* Visualize the keypoints on the image and display the result.

![FAST (Features from Accelerated Segment Test) Keypoint Detection](https://github.com/user-attachments/assets/f268b0e8-26c7-4141-aec7-6d165c7d4b5d)

Keypoint Detection algorithm of FAST features in an accelerated segment test was applied to a night image of a building. The total number of keypoints detected was 21,310. FAST features are applied to rapidly detect corner-like features by testing the intensity of pixels around a candidate point in a circular pattern. In this case, the threshold was set as 10, so it found only those points where the pixel intensity change exceeded this value, making those regions contain key features. The parameter `nonmaxSuppression: True` ensures only the strongest keypoints are kept and weaker responses around a stronger one are suppressed to avoid multiple detections in close proximity. The `neighborhood: 2` parameter describes the way the algorithm calculates the pixels around which contribute to corner responses.

The building image taken in nighttime, because of artificial lighting, shadows, and structural edges, boasts sharp contrast in intensity that is well suited for a keypoint detection algorithm. Lit areas and their corresponding dark regions, as well as the repeating geometric patterns from the structure of the building, make corner features hard to miss. FAST is very well suited to such environments, as it can highly efficiently detect keypoints in images of high contrast with a rather low cost in the form of computations that render the algorithm highly powerful when applications demand real-time processing. The large number of keypoints detected means the algorithm found many distinctive points that could be useful for tasks like matching and tracking images or for 3D reconstruction when light is low.

### Key Points
* FAST is designed to be computationally efficient and quick in detecting keypoints.
* It is often used in real-time applications like robotics and mobile vision.

## Exercise 4: Feature Matching using ORB and FLANN
### Task
Use ORB descriptors to find and match features between two images using FLANN-based matching.
* Load two images of your choice.
* Extract keypoints and descriptors using ORB.
* Match features between the two images using the FLANN matcher.
* Display the matched features.

![Feature Matching using ORB and FLANN](https://github.com/user-attachments/assets/c1241d43-9fbe-4628-a900-eaaa9d788c25)

On the two images of the flower captured from perspectives, feature matching was done using the ORB (Oriented FAST and Rotated BRIEF) combined with FLANN (Fast Library for Approximate Nearest Neighbors). The ORB detector would find keypoints to a number of 500 while in the case of the FLANN matcher, there would be 1,000 matches between these keypoints. ORB is an efficient algorithm for feature detection and description and insusceptible to scale and rotation, which makes it ideal for real-time applications. It detects keypoints and computes descriptors of these points, which encode local appearance of the image around each keypoint. Then FLANN is utilized in order to match descriptors by finding the nearest neighbors between the two images even if they have differences such as scale, rotation, or perspective.

The pictures of the flower taken from different angles are appropriate for this approach because flowers generally have unique textures, edges, and patterns that can produce numerous unique keypoints. The 1,000 matches suggest that although the viewing angle has been changed, the two images have a significant number of corresponding features. It means that ORB with FLANN excels in the identification of similar structures from different viewpoints, thus making it a proper solution for object recognition, 3D reconstruction, or image stitching. Good performance in feature matching reflects the robustness of the algorithm to viewpoint changes and led to the suitability of the method for employing distinguishable objects with rich details to match.

### Key Points
* **ORB** is fast and efficient, making it suitable for resource-constrained environments.
* **FLANN** (Fast Library for Approximate Nearest Neighbors) speeds up the matching process, making it ideal for large datasets.

## Exercise 5: Image Segmentation using Watershed Algorithm
### Task
The Watershed algorithm segments an image into distinct regions.
* Load an image.
* Apply a threshold to convert the image to binary.
* Apply the Watershed algorithm to segment the image into regions.
* Visualize and display the segmented regions.

![Image Segmentation using Watershed Algorithm](https://github.com/user-attachments/assets/6de00bd9-343f-458c-8832-0309ab14511b)

An image of candies was subjected to watershed algorithm-based image segmentation. This is an algorithm for the segmentation of an image with overlapping or touching objects. The nature of pixel intensity values is treated as the topographic surface by the algorithm, in which the peaks represent the high-intensity regions and the valleys correspond to low-intensity regions. By flooding the image from its local minima, the algorithm can separate regions by identifying boundaries, much like watersheds divide land areas. The input image shown here above left to illustrate segmentation using candies, which are good for testing the segmentation process because of clear contrast in the area between candy edges and their surrounding regions.

The results include the following stages: a thresholded image (right top), segmented regions (left bottom), and a final image with segmentation boundaries (right bottom). The thresholded image is a binarization of the original, it isolates all the regions of interest within which the algorithm will operate. Segmented regions demonstrate how candies were categorized into different groups according to color intensity, while the final image depicts the boundary of the region with red lines in order to distinguish the regions.

The above image is suitable for Watershed Algorithm as candies are well-defined with strong edges facilitating the algorithm to distinguish between boundaries of objects in contact or overlap. Bright colors as well as varied intensities further help the algorithm to produce a fine segmentation. The Watershed method is highly efficient for such a task, in where objects are composed of several overlapping objects of comparable intensities, since the method will produce unambiguous separations, based both on intensity gradients and object edges.

### Key Points
* Image segmentation is crucial for object detection and recognition.
* The Watershed algorithm is especially useful for separating overlapping objects.
