# Feature Extraction Methods
[Feature extraction](https://domino.ai/data-science-dictionary/feature-extraction) is a technique used to transform raw data into a set of measurable properties or features that can be used for various tasks, such as classification, prediction, and clustering. The primary goal is to reduce the complexity of the data while retaining essential information that contributes to the performance of machine learning models. [^1]

In the context of computer vision (CV), it refers to the process of identifying and isolating relevant characteristics or features from raw image data. This step is crucial for enabling machine learning algorithms to effectively analyze and interpret visual information.

## Task 1: SIFT Feature Extraction
### Definition
SIFT, or **Scale-Invariant Feature Transform**, is a robust algorithm used in computer vision for detecting and describing local features in images. Developed by David Lowe in 1999, SIFT is particularly valuable for tasks that require matching key points between different images, regardless of changes in scale, rotation, or viewpoint. [^2] The Scale-Invariant Feature Transform (SIFT) is applied to an image to detect and describe local features. SIFT extracts key points based on scale and orientation invariance.
### Observations
The SIFT algorithm was able to detect distinct key points in high-contrast areas of the image. It was particularly effective in extracting features from textured regions like edges or corners.
### Results
The output consisted of a set of key points and descriptors that describe the local features around these points. It specifically results into 938 key points detected. These key points were scale and rotation invariant.

![Task 1: SIFT Feature Extraction](https://github.com/user-attachments/assets/678dd5df-a7d3-4e11-b07d-f40c2191a686)

## Task 2: SURF Feature Extraction
### Definition
SURF, or **Speeded Up Robust Features**, is a feature detection and description algorithm designed to improve upon the capabilities of SIFT (Scale-Invariant Feature Transform) while significantly increasing computational speed. Developed by Herbert Bay and colleagues in 2006, SURF is particularly effective in real-time applications such as object recognition and image matching. [^3] It uses integral images and box filters to approximate key points and descriptors more quickly, focusing on blob-like structures.
### Observations
SURF detected a significantly higher number of key points than SIFT, identifying 4000 key points in the image. These points were detected quickly, demonstrating the efficiency of the algorithm. However, while the number of detected features was higher, some of them appeared less distinctive compared to the features detected by SIFT.
### Results
SURF successfully identified 4000 key points, offering a much larger set of features compared to SIFT. However, the higher number of key points came with a trade-off in descriptor quality, especially in fine-detail areas.

![Task 2: SURF Feature Extraction](https://github.com/user-attachments/assets/1fc09937-8b76-40f1-8cb2-9b82fbccd53e)

## Task 3: ORB Feature Extraction
### Definition
ORB, or **Oriented FAST and Rotated BRIEF**, is a feature detection and description algorithm introduced by Ethan Rublee and colleagues in 2011. It is designed to provide a fast and efficient alternative to other feature extraction methods like SIFT and SURF, while being free from patent restrictions. [^4] Oriented FAST and Rotated BRIEF (ORB) is a binary descriptor-based algorithm that combines FAST key point detection with BRIEF descriptors. ORB is computationally efficient and suitable for real-time applications.
### Observations
ORB detected 500 key points, which is significantly fewer than both SIFT and SURF. The key points were concentrated in areas of high contrast, but the algorithm struggled to identify features in smooth or low-texture regions. Although ORB was computationally efficient and completed feature extraction quickly, the detected points were less precise than those identified by SIFT, particularly in scenes with complex textures.
### Results
ORB produced a set of 500 key points, fewer than both SIFT and SURF. While these binary descriptors were efficient, their precision was lower, which may affect the accuracy of subsequent matching tasks.

![Task 3: ORB Feature Extraction](https://github.com/user-attachments/assets/b674bf43-0721-4fc8-8d38-bc27e845341d)

## Task 4: Feature Matching using SIFT
### Definition
Feature matching using SIFT (Scale-Invariant Feature Transform) is a critical process in computer vision that involves identifying and correlating keypoints between different images based on their distinctive features. This technique is particularly effective for tasks such as object recognition, image stitching, and 3D reconstruction.
### Observations
258 successful matches were found between the images. The matching process was accurate, with minimal outliers, due to the SIFT descriptors. 
### Results
The feature matching resulted in 258 successful matches which demonstrates SIFT’s robustness in accurately identifying corresponding points between the images. These matches were largely consistent, even in the presence of variations in scale and orientation.

![Task 4: Feature Matching using SIFT](https://github.com/user-attachments/assets/479691ec-6497-4945-84d3-324c0029e745)

## Task 5: Image Stitching Using Homography
### Definition
Image stitching using homography is a technique in computer vision that involves combining multiple images into a single panoramic image by aligning them based on their overlapping regions. This process relies on estimating a homography matrix, which describes the geometric transformation needed to align the images.

SIFT was employed to detect key points and extract descriptors from two images. After extracting the descriptors, a Brute-Force Matcher (BFMatcher) was used to compare the descriptors and find corresponding key points between the two images. To improve the quality of the matches, a ratio test was applied to filter out less reliable matches. The matched points were then used to compute a homography matrix with RANSAC to ensure that only the best matches contributed to the alignment process. Finally, this homography matrix was applied to warp one image onto the other for alignment and image stitching.
### Observations
SIFT and BFMatcher effectively detected and matched key points between the two images, and the ratio test filtered out many false matches. The RANSAC algorithm further improved the robustness of the homography computation by eliminating outlier matches. The warping process aligned the two images well, although minor distortions appeared near the edges, particularly in areas with fewer key points or repetitive textures. Despite these small imperfections, the overall alignment was smooth, and the images were stitched together with high precision, especially in regions with distinct features.
### Results
The two images were successfully aligned using SIFT, BFMatcher, and homography. The stitching was accurate, and the images were warped to fit together and the image was unrecognizable and incomprehensible.

![Task 5: Image Stitching Using Homography](https://github.com/user-attachments/assets/508f3dd8-d810-4b71-a1bb-3fce062b06eb)

## Task 6: Combining SIFT and ORB

### Definition
Combining SIFT (Scale-Invariant Feature Transform) and ORB (Oriented FAST and Rotated BRIEF) involves leveraging the strengths of both algorithms to enhance feature detection, extraction, and matching in computer vision tasks. Each algorithm has unique advantages: SIFT is known for its high accuracy and robustness against various transformations, while ORB is recognized for its speed and efficiency.
### Observations
SIFT detects more key points and generates a higher number of matches between the two images, particularly focusing on prominent facial features such as the eyes, nose, and mouth. The lines between the images indicate the correspondence of these points, and they are mostly consistent across the faces. ORB generates fewer matches compared to SIFT, and some of the correspondences are less precise. The matches are more scattered across the face, and a few mismatches appear, particularly along the edges of the image and hairline.
### Results
**SIFT** detected more matches (258) with higher precision, particularly in areas with high-contrast and complex textures like facial features. **ORB** detected fewer matches (89), and while it provided some useful correspondences, the overall accuracy was lower, with some mismatches occurring in less distinctive regions.

![Task 6: Combining SIFT and ORB](https://github.com/user-attachments/assets/1add2e86-21ec-415a-85ac-1dbe11167d4f)

## References
[^1]: Domino Data Lab. (2024). _What is Feature Extraction?_ https://domino.ai/data-science-dictionary/feature-extraction
[^2]: AishwaryaSingh. (2019). _What is SIFT (Scale Invariant Feature Transform) Algorithm?_ Analytics Vidhya. https://www.analyticsvidhya.com/blog/2019/10/detailed-guide-powerful-sift-technique-image-matching-python/
[^3]: Bay, H., Ess, A., Tuytelaars, T., & Van Gool, L. (2006). _Speeded-Up Robust Features (SURF)_. Computer Vision and Image Understanding. https://www.sciencedirect.com/science/article/abs/pii/S1077314207001555
[^4]: Geeks for Geeks. (2024). _Feature matching using ORB algorithm in PythonOpenCV_. https://www.geeksforgeeks.org/feature-matching-using-orb-algorithm-in-python-opencv/
