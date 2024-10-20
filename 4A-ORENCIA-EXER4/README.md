# Exercise 1: HOG (Histogram of Oriented Gradients) Object Detection
## Task
HOG is a feature descriptor widely used for object detection, particularly for human detection. In this exercise, you will:
* Load an image containing a person or an object.
* Convert the image to grayscale.
* Apply the HOG descriptor to extract features.
* Visualize the gradient orientations on the image.
* Implement a simple object detector using HOG features.

![HOG Object Detection](https://github.com/user-attachments/assets/59e95e8d-7d9b-447d-a74f-4428fb0b8697)

The image result above comes from a code that first reads an image from a particular path, then proceeds with pre-processing. It crops the image to a square shape by finding its minimum dimension, along with further conversion into a grayscale intensity image for further analysis.

The HOG descriptor is then applied on the grayscale image. It extracts features and produces a HOG visualization. The visualization makes it easier to identify gradient orientations in an image that may contribute to the eventual recognition of a pattern or shape.

At the last stage, all the results are shown in a single figure containing three subplots in such an order of original image, grayscale image, and HOG visualization. With this layout, it is possible to be able to look at how each of the various steps in the processing of an image alters it.
## Key Points
* HOG focuses on the structure of objects through gradients.
* Useful for detecting humans and general object recognition

# Exercise 2: YOLO (You Only Look Once) Object Detection
## Task
YOLO is a deep learning-based object detection method. In this exercise, you will:
* Load a pre-trained YOLO model using TensorFlow.
* Feed an image to the YOLO model for object detection.
* Visualize the bounding boxes and class labels on the detected objects in the image.
* Test the model on multiple images to observe its performance.

![YOLO Object Detection](https://github.com/user-attachments/assets/f51d5aac-86e2-4750-8ec6-90afda7d372a)

The result above comes from a written code that uses YOLO with OpenCV to execute an object detection pipeline. This code loads the weights and configuration of the YOLO model along with class names from the COCO dataset. The code then goes through a sequence of images cropped to a 3:2 aspect ratio. It's the result of a predefined function that performs a center crop. With each of the cropped images, a blob is created for sizing down to 416x416 pixels for proper input into the YOLO network.

Next, the code runs the YOLO model to detect objects on each image, and the code selects which object detected to include in its output - filtering based on confidence score and implementing the non-max suppression to remove overlapping bounding boxes. It overlays the bounding boxes and labels on the images while adjusting the font size and color to make it more visible. After processing, images are transformed from BGR format to RGB as required for display using Matplotlib library.

In the output of object detection, there exists a number of objects with their related confidence scores. For example in the first image, it is clearly identified several persons with high confidence scores while it incorrectly has labelled the cup as a phone. Similar results are delivered in other images also wherein it has correctly classified persons in maximum instances though it does misclassify such as classifying a dining object and a book with lower confidence scores. Overall, the YOLO model gives great performance in the detection of people across all images while producing some occasional errors in classification.
## Key Points
* YOLO is fast and suitable for real-time object detection.
* It performs detection in a single pass, making it efficient for complex scenes.

# Exercise 3: SSD (Single Shot MultiBox Detector) with TensorFlow
## Task
SSD is a real-time object detection method. For this exercise:
* Load an image of your choice.
* Utilize the TensorFlow Object Detection API to apply the SSD model.
* Detect objects within the image and draw bounding boxes around them.
* Compare the results with those obtained from the YOLO model.

![SSD with Tensorflow](https://github.com/user-attachments/assets/9d79b283-9624-484e-843c-b9488dcf268d)

The following result above comes from a code configures an object detection pipeline using the SSD MobileNet V2 model from TensorFlow Hub. The process begins with importing the pre-trained model and then defines a list of the image files to be analyzed. Each image reads and crops, utilizing a function that centers it into a 3:2 aspect ratio so that all the inputs are uniform.

After the images are cropped, they undergo a conversion to tensors and are then used to input into the TensorFlow model. It iterates over the tensor, detecting objects and returning detection boxes, confidence scores, and class identifiers for every detected object. A threshold of 0.5 is set for confidence scores to eliminate detections where the detection confidence is less than 0.5.

Bounding boxes are drawn around the detected areas in the images along with labels mentioning the confidence score for all detected objects that are above a given confidence threshold. The processed images are then converted from BGR to RGB format and prepared for visualization using Matplotlib.

Different confidence scores are shown in each of the images. For instance, in the first image several objects are detected with confidence scores between 0.63 and 0.70. The same results can be obtained for other images. The second shows even higher scores of 0.80 and 0.81. Scores in the third to fifth images also explain successful detections since most confidence scores remain above 0.70. It proves that SSD MobileNet V2 is really good at detecting objects in a given set of images but shows all the labels generically as "Object" without actually showing any particular class of object.
## Key Points
* SSD is efficient in terms of speed and accuracy.
* Ideal for applications requiring both speed and moderate precision.

# Exercise 4: Traditional vs. Deep Learning Object Detection Comparison
## Task
Compare traditional object detection (e.g., HOG-SVM) with deep learning-based methods (YOLO, SSD):
* Implement HOG-SVM and either YOLO or SSD for the same dataset.
* Compare their performances in terms of accuracy and speed.
* Document the advantages and disadvantages of each method.

The following code that compares two object detection techniques which are HOG-SVM and YOLO is written. The technique HOG-SVM first extracts features from the input images. The code converts the images to grayscale, resizes them, and utilizes the HOG technique, which produces a feature vector. It now uses that vector as input to the SVM model for the purpose of prediction to classify whether the target object is presented in the image or not.

On the other hand, the YOLO approach depends on a pre-trained YOLO model to classify objects of interest right from the image. It checks whether the detected objects fall into a class of interest thus making it possible to do real-time detection in an efficient manner. The script consists of a training phase for the HOG-SVM model, which takes on a labeled dataset of images. This dataset contains the presence or absence of the object of interest labeled to be a person, among other things.

During the evaluation phase, the two models are tested on the same dataset of images. The script write down the result of their predictions and compute metrics on the accuracy of each method. Additionally, the total time taken to process by each approach allow for a direct comparison of the two models. Thus, the output shows the accuracy and efficiency of the HOG-SVM and YOLO methods, which could give a better understanding of which performs better in object detection tasks.

* `HOG-SVM Accuracy: 0.6667`
* `YOLO Accuracy: 1.0000`
* `HOG-SVM Total Time: 0.0206 seconds`
* `YOLO Total Time: 1.1926 seconds`

The results of such experiments also outline a pretty relevant difference between the two object detection methods under investigation: HOG-SVM and YOLO. For example, the accuracy achieved by the HOG-SVM model in the case at hand was 0.6667, meaning that it correctly detected the occurrence of objects in about two thirds of its results. In contrast, the model of the YOLO resulted in the perfect accuracy of 1.0000 and correctly detected all of the target objects used in the test dataset in actuality. This drastic difference clearly shows that YOLO is far more accurate in identifying the objects of interest in these test images.

Although YOLO is more accurate, it consumes more processing time. The proof lies in its total processing time on each image, with a difference of 1.1926 seconds from HOG-SVM's mere 0.0206 seconds. This means, even though YOLO detection is more accurate, it is slower and not suitable for real-time applications. While, HOG-SVM approach may not be so accurate but is much faster. So, in such scenarios, where execution speed is the first priority, but where a little distortion in accuracy can be tolerated, this approach applies.

The advantages of HOG-SVM is that it is faster than other methods, and hence suitable for real-time applications; simpler than other methods; however, may not be as precise as deep learning-based methods like YOLO, especially when objects are partially occluded from the camera and in complex scenes. Does not generalize well to different datasets or if the appearance changes.

On the other hand, YOLO is more accurate in object detection hence appropriate when it comes to applications where reliable and accurate detections are necessary. It can apply various real-time applications despite taking more processing time compared to HOG-SVM. The YOLO is computationally intensive, therefore is more likely to take a longer time to process, which might be its limitation in places where the output needs to come immediately. Complexity is another weakness, since models of YOLO are complex to configure, and thus the capability needed to compute these is more comprehensive, thereby less accessible for minor applications.

In conclusion, the choice between HOG-SVM and YOLO depends solely on the particular application's requirements in terms of accuracy tolerance versus speed of processing. For demanding applications where accuracy matters the most, it would be better off choosing YOLO, while HOG-SVM is preferred in applications that are rapid detection-oriented over accuracy.
## Key Points
* Traditional methods may perform better in resource-constrained environments.
* Deep learning methods are generally more accurate but require more computational power.
