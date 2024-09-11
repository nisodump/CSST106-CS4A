# Machine Problem No. 1: Exploring the Role of Computer Vision and Image Processing in AI

https://github.com/user-attachments/assets/f8e89efa-216b-4ab1-8adf-fa10b1260d3d

---
## Objective
Understand the importance of computer vision and image processing in Artificial Intelligence (AI) and explore how these technologies enable AI systems to analyze and interpret visual data.

---
## Research and Comprehend
### Introduction to Computer Vision
[Computer vision](https://en.wikipedia.org/wiki/Computer_vision) is a field of [artificial intelligence (AI)](https://en.wikipedia.org/wiki/Artificial_intelligence) that enables computers to interpret and understand visual information from the world, similar to how humans perceive and process images. It encompasses a variety of techniques and technologies aimed at enabling machines to "see" and make decisions based on visual data. [^1]
#### Basic Concepts of Computer Vision
1. **Visual Data Processing**
- At its core, computer vision involves teaching machines to recognize patterns and objects in images and videos. This is achieved through algorithms that analyze visual data, transforming it into actionable insights. The process mimics human visual perception, where the brain interprets light signals into recognizable images and patterns. [^2]
2. **Machine Learning and Deep Learning**
- Modern computer vision heavily relies on machine learning and deep learning techniques. These methods allow AI systems to learn from large datasets of images, improving their ability to recognize and categorize objects over time. Neural networks, particularly convolutional neural networks (CNNs), are commonly used to process visual data, enabling tasks such as image classification and object detection. [^3]
3. **Image Recognition and Object Detection**
- Image recognition refers to identifying and classifying objects within an image (e.g., distinguishing between a cat and a dog). Object detection refers to locating and identifying multiple objects within an image or video frame, such as detecting pedestrians in self-driving car applications. [^4]
4. **Facial Recognition**
- Facial recognition technology uses computer vision to identify and verify individuals based on facial features. This technology is widely used in security systems and social media platforms for tagging and authentication purposes. [^5]
5. **Scene Understanding**
- Beyond recognizing individual objects, AI systems can analyze entire scenes to understand context. This includes interpreting actions and interactions within a visual environment, which is crucial for applications like autonomous driving and surveillance. [^4]
6. **Anomaly Detection**
- Computer vision can also identify unusual patterns or objects in visual data, which is useful in security (e.g., detecting intruders) and quality control in manufacturing (e.g., spotting defects in products). [^6]
#### Role of Image Processing in AI
[Image processing](https://www.simplilearn.com/image-processing-article) plays a pivotal role in artificial intelligence (AI) by enabling systems to effectively enhance, manipulate, and analyze visual data. This capability is essential for various applications, ranging from consumer products to advanced technologies in fields such as healthcare and autonomous vehicles.
1. **Enhancing Image Quality**
- AI-driven image processing tools utilize machine learning algorithms to enhance image quality automatically. Techniques such as noise reduction, color correction, and sharpening allow for the transformation of low-quality images into high-definition visuals [^7]. For example, AI can intelligently remove noise from photos taken in low-light conditions while preserving essential details, thus improving clarity and overall quality. 
2. **Object Recognition and Analysis**
- Image processing enables AI systems to recognize and classify objects within images. This capability is fundamental for applications like facial recognition, autonomous driving, and medical imaging. For instance, self-driving cars rely heavily on image processing to interpret their surroundings, ensuring safe navigation by recognizing pedestrians, traffic signs, and other vehicles. [^4]
3. **Automation of Editing Processes**
- AI significantly speeds up complex image editing tasks that traditionally required manual intervention. Features such as automatic object removal and background replacement can be executed in seconds, which previously might have taken hours of work. This efficiency not only enhances productivity but also allows creators to focus more on the creative aspects of their work rather than repetitive editing tasks. [^8]
4. **Creative Enhancement**
- AI image processing tools also facilitate creative expression through features like style transfer, which applies the artistic style of one image to another. This opens new avenues for artistic experimentation, allowing users to transform ordinary photos into artwork. The integration of AI in creative processes fosters innovation and enhances the quality of visual content produced across various media. [^7]
### Overview of Image Processing Techniques
#### Core Techniques in Image Processing
##### Filtering
[Filtering](https://www.scaler.com/topics/filtering-in-image-processing/) is a technique used to enhance or suppress certain features of an image. It involves applying mathematical operations to the pixel values of an image to achieve desired effects, such as noise reduction or sharpening.
- **How it helps AI**: Filtering improves image quality, making it easier for AI systems to identify and classify objects. For instance, Gaussian filters can smooth images, reducing noise and allowing for clearer feature extraction, which is critical for tasks like facial recognition and object detection. [^9]
##### Edge Detection
[Edge detection](https://blog.roboflow.com/edge-detection/) is a technique used to identify the boundaries of objects within an image. This is typically achieved through algorithms that highlight significant changes in intensity or color, marking the edges of objects.
- **How it helps AI**: By detecting edges, AI systems can better understand the structure and shape of objects. This is particularly important in applications like autonomous driving, where recognizing the edges of roads, vehicles, and pedestrians is essential for safe navigation. [^10]
##### Segmentation
[Segmentation](https://www.ibm.com/topics/image-segmentation) involves partitioning an image into multiple segments or regions, each representing different objects or parts of the image. This can be done using various methods, such as thresholding, clustering, or deep learning techniques.
- **How it helps AI**: Segmentation allows AI systems to analyze specific regions of an image independently, facilitating more accurate object recognition and classification. For example, in medical imaging, segmentation can isolate tumors or other structures within scans, enabling more precise diagnostics and treatment planning. [^11]
---
## Hands-On Exploration
### Case Study Selection
The use of artificial intelligence (AI) in medical image analysis has significantly advanced, particularly in detecting diseases such as pneumonia from chest radiographs. A notable application is the development of privacy-preserving AI models that uses image processing techniques to accurately classify X-ray images while protecting patient data. In this context, Convolutional Neural Networks (CNNs) are employed to identify pneumonia by automatically detecting visual features such as lung opacity. However, a major concern arises from the fact that deep learning models are prone to privacy risks, as they can memorize specific data points from the training dataset, potentially exposing sensitive information. To address this, Differential Privacy (DP) is integrated into the model training process, ensuring that individual patients' data remains secure.

The problem this approach addresses is twofold: the need for accurate medical image analysis, such as pneumonia detection in radiographs, and the protection of sensitive patient data from being leaked or reverse-engineered from AI models. Deep learning models, though powerful, pose privacy risks due to their capacity to memorize training data, which is particularly concerning in the medical field. Differential Privacy provides a solution by adding noise during model training, preventing the exposure of individual patient information while still enabling the model to generalize well and make accurate predictions.
#### Image Processing in Pneumonia Detection Using CNNs
In this process, chest X-ray images (radiographs) are collected, typically from a public dataset that includes images of patients with pneumonia as well as normal (no findings) cases. These images undergo preprocessing steps, including:
- **Rescaling or Normalization**: Ensuring the pixel values are on a uniform scale (e.g., between 0 and 1) to make the model training stable.
- **Augmentation**: Techniques like rotation, flipping, or zooming can be applied to increase data variety, improving model robustness.
- **Cropping and Resizing**: Standardizing image size to match the input requirements of the CNN.

CNNs are highly effective in image recognition tasks, as they automatically detect key features from the images through multiple layers of convolutional filters. In this context:
- **Convolutional Layers**: Extract features such as edges, textures, or patterns within the chest X-ray images that may indicate pneumonia.
- **Pooling Layers**: Reduce dimensionality, retaining essential features while cutting down on computational complexity.
- **Fully Connected Layers**: Combine extracted features to classify images as either pneumonia-positive or negative. CNNs have shown high effectiveness in recognizing pneumonia in radiographs by identifying visual markers like lung opacity or consolidation, which are common in pneumonia-affected lungs.
#### Differential Privacy in Medical Image Analysis
**Differential Privacy (DP)** is used to ensure that the deep learning model does not inadvertently reveal sensitive information about individual patients. This is crucial because large deep learning models, especially CNNs with millions of parameters, are at risk of memorizing specific data points, including patient information from medical images.

In the DP-enabled training, privacy-preserving techniques are applied during model training, ensuring that no single patient's data can be reverse-engineered from the trained model. The key techniques include:
- **Noise Injection**: Adding calibrated noise to the gradients during training prevents the model from overfitting or memorizing individual images, ensuring that even if the model parameters are leaked, individual patient data remains anonymous.
- **Clipping of Gradients**: Gradient clipping ensures that each individual’s data contribution to the model’s training remains limited, preventing disproportionately large updates from any single data point, which could potentially reveal sensitive information. These techniques ensure that the model adheres to DP guarantees while maintaining a high level of accuracy in detecting pneumonia.
#### Effectiveness of Image Processing Techniques in Privacy-Preserving Models
- **Baseline (Non-Differentially Private) Model**: Typically, a CNN without DP training performs slightly better in terms of classification accuracy, as it fully utilizes the information in the training data. This non-private model serves as a baseline for comparison.
- **Differentially Private Model**: While DP introduces a slight trade-off in model performance due to the added noise and constraints on gradient updates, recent advancements ensure that these models can still perform well. The performance gap between DP models and non-DP models is often minimal, especially when privacy-preserving techniques are carefully tuned.
#### Effectiveness in Solving Visual Problems
The use of **CNNs** in medical image analysis, particularly for pneumonia detection in radiographs, is highly effective due to their ability to learn from complex visual patterns. When combined with **Differential Privacy**, these models balance the need for accurate medical diagnosis with robust privacy protections. Although DP introduces some performance trade-offs, modern implementations ensure that privacy-preserving models remain clinically useful.
### Implementation Creation
Differential Privacy is considered the most robust method for safeguarding individuals' data when generating and publishing statistical analyses. It provides a quantifiable privacy assurance to each data subject, far exceeding the protection offered by commonly used techniques like data anonymization. 

Contemporary deep learning models, with their vast capacity often encompassing millions or even billions of trainable parameters, are particularly vulnerable to privacy risks. These models have a propensity to memorize individual data points or entire datasets, which is undesirable not only from a privacy standpoint but also because the primary objective of machine learning is to train models that can learn general patterns to make accurate predictions on new data, rather than simply memorizing the training data.

The code for the Jupyter notebook below was developed and tested using an Azure ML NC12v3 GPU [compute instance](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-manage-compute-instance?tabs=python). It is from the [Privacy Preserving Medical Image Analysis](https://github.com/Azure/medical-imaging/blob/main/notebooks/5-diff-privacy-pneumonia-detection.ipynb) made by Microsoft AI Rangers.
#### Setups
1. **Installs and imports**
```python
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
%config InlineBackend.figure_format = 'retina'

import time
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from opacus import PrivacyEngine

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score

from utils.demoutils import dptrain, plot_learning_curve, to_categorical, print_metrics, predict_loader
```

2. **Set Global Variables**
```python
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
print(device)

# Source of compressed image dataset
ds_zip_source = 'datasets/chest-pat-2class-tvt.zip'
# Base directory to unzip the dataset
base_dir = '/home/azureuser/chest/'

seed = 1

img_height, img_width, channels = 224, 224, 1
img_mean = 0.4818
img_std = 0.2357
```

#### Acquire and prepare x-ray images
1. **Create dataset on local Compute Instance filesystem**
```python
extract = False

if extract == True:
    from zipfile import ZipFile
    
    with ZipFile(ds_zip_source, 'r') as zipObj:
        zipObj.extractall(base_dir)

torch.manual_seed(seed)
transform_train = transforms.Compose([transforms.Grayscale(),
                                transforms.Resize((img_height, img_width)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean = (img_mean,), std = (img_std,))    
                               ])

transform_val = transforms.Compose([transforms.Grayscale(),
                                transforms.Resize((img_height, img_width)),
                                transforms.ToTensor(),   
                                transforms.Normalize(mean = (img_mean,), std = (img_std,))    
                               ])

training_dataset = datasets.ImageFolder(root = base_dir + 'train/', transform = transform_train)
validation_dataset = datasets.ImageFolder(root = base_dir + 'val/', transform = transform_val)
test_dataset = datasets.ImageFolder(root = base_dir + 'val/', transform = transform_val)

training_loader = torch.utils.data.DataLoader(dataset = training_dataset, batch_size = 32, shuffle = True, drop_last=True)
validation_loader = torch.utils.data.DataLoader(dataset = validation_dataset, batch_size = 32, shuffle = False, drop_last=True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = 100, shuffle = False)
```

2. **Show examples of both classes**
```python
def im_convert(tensor):
    image = tensor.clone().detach().numpy()
    image = image.transpose(1, 2, 0) # convert shape to WHC
    image = image * np.array (img_std,) + np.array(img_mean,) # revert normalization
    image = image.clip(0, 1)
    return image

dataiter = iter(training_loader)
images, labels = dataiter.next()

fig = plt.figure(figsize = (18, 8))
for idx in np.arange(12):
    ax = fig.add_subplot(2, 6, idx + 1, xticks = [], yticks = [])
    im = im_convert(images[idx]).reshape(img_height, img_width)
    plt.imshow(im , cmap = 'gray')
    ax.set_title(training_dataset.classes[labels[idx].item()])

plt.tight_layout()
plt.show()
```
![Image](https://i.imgur.com/KceI72S.jpeg)

#### Develop a Convolutional Neural Network (CNN) for image classification
```python
class Cnn(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels = 1,out_channels = 32, kernel_size = 3, stride = 1, padding=  1)
        self.conv2 = nn.Conv2d(in_channels = 32,out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(in_channels = 64,out_channels = 128, kernel_size = 3, stride = 1, padding = 1)
             
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.3)
                
        self.fc1 = nn.Linear(28*28*128, 256)
        self.fc2 = nn.Linear(256, 2)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))  # 224 x 224 x 32
        x = F.max_pool2d(x, 2, 2)  # 112 x 112 x 32
        x = F.relu(self.conv2(x))  # 112 x 112 x 64
        x = F.max_pool2d(x, 2, 2)  # 56 x 56 x 64
        x = self.dropout1(x)
        x = F.relu(self.conv3(x))  # 56 x 56 x 128
        x = F.max_pool2d(x, 2, 2)  # 28 x 28 x 128
        x = self.dropout2(x)       
        x = x.view(-1, 28*28*128) # 100.352
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

#### Train a standard (non differentially private) model as a baseline
```python
model = Cnn().to(device)

torch.manual_seed(seed)

history = dptrain(model = model,
                  optimizer = torch.optim.Adam(model.parameters(), lr = 0.0007),
                  loss_fn = nn.CrossEntropyLoss(),
                  train_dl = training_loader,
                  val_dl = validation_loader,
                  epochs = 15,
                  device = device,
                  private_training = False)
```

1. **Review training progress**
```python
fig = plot_learning_curve(history)
plt.show()
```
![Image](https://i.imgur.com/4BMM9ji.png)
```python
torch.save(model, 'models/pneumonia-nonpriv.pth')
```

2. **Evaluate the model**
```python
model = torch.load('models/pneumonia-nonpriv.pth')
```
```python
y_true, y_pred, y_probs = predict_loader(model, test_loader, device)

y_true = y_true.cpu()
y_pred = y_pred.cpu()
y_probs = y_probs.cpu()

y_true_oh = to_categorical(y_true, num_classes = 2)
y_pred_oh = to_categorical(y_pred, num_classes = 2)

print_metrics(y_true_oh, y_pred_oh, y_probs, test_dataset.classes)
```

#### Use Differential Privacy to train a privacy preserving model
```python
data_loader = torch.utils.data.DataLoader(dataset = training_dataset, batch_size = 8) #, shuffle = True, drop_last=True)

model = Cnn().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = 0.0007)

privacy_engine = PrivacyEngine()

model, optimizer, data_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=data_loader,
    noise_multiplier=0.65,
    max_grad_norm=2.0)
```
```python
torch.manual_seed(0)

history = dptrain(model = model,
                  optimizer = optimizer,
                  loss_fn = nn.CrossEntropyLoss(),
                  train_dl = data_loader,
                  val_dl = validation_loader,
                  epochs = 15,
                  device = device,
                  private_training = True,
                  privacy_engine = privacy_engine,
                  target_delta = 1/5000)
```
```python
epochs = range(1, len(history['acc']) + 1)

fig, ax1 = plt.subplots(figsize=(12, 6))

plt.plot(history['loss'], 'red', label='Training loss')
plt.plot(history['val_loss'], 'green', label='Validation loss')
plt.legend(loc='center right')
plt.title('Training / validation loss and epsilon')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()

ax2 = ax1.twinx()

color = 'tab:blue'
ax2.set_ylabel('Epsilon', color=color)
ax2.plot(history['epsilon'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.show()
```
![Image](https://i.imgur.com/A4XdKME.png)
```python
plt.clf()

epochs = range(1, len(history['acc']) + 1)

fig, ax1 = plt.subplots(figsize=(12, 6))

plt.plot(history['acc'], 'orange', label='Training accuracy')
plt.plot(history['val_acc'], 'blue', label='Validation accuracy')
plt.legend(loc='center right')
plt.title('Training / validation accuracy and epsilon')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid()

ax2 = ax1.twinx()

color = 'tab:blue'
ax2.set_ylabel('Epsilon', color=color)
ax2.plot(history['epsilon'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.show()
```

1. **Option to save / load the model**
```python
torch.save(model, 'models/pneumonia-dp.pth')
model = torch.load('models/pneumonia-dp.pth')
```

2. **Evaluate accuracy on test set**
```python
y_true, y_pred, y_probs = predict_loader(model, test_loader, device)

y_true = y_true.cpu()
y_pred = y_pred.cpu()
y_probs = y_probs.cpu()

y_true_oh = to_categorical(y_true, num_classes = 2)
y_pred_oh = to_categorical(y_pred, num_classes = 2)

print_metrics(y_true_oh, y_pred_oh, y_probs, test_dataset.classes)
```
#### Compare the performance of both models
```python
# Get y_true from testloader and predicted propabilities from non-private model:
np_model = model = torch.load('models/pneumonia-nonpriv.pth')
y_true, _, y_probs_np = predict_loader(np_model, test_loader, device)
y_true = y_true.cpu()
y_true_oh = to_categorical(y_true, num_classes = 2)
y_probs_np = y_probs_np.cpu()

# Get predicted propabilities from DP model:
dp_model = model = torch.load('models/pneumonia-dp.pth')
_, _, y_probs_dp = predict_loader(dp_model, test_loader, device)
y_probs_dp = y_probs_dp.cpu()
```
```python
np_fpr, np_tpr, np_thresholds = roc_curve(y_true_oh[:,0], y_probs_np[:,0])
dp_fpr, dp_tpr, dp_thresholds = roc_curve(y_true_oh[:,0], y_probs_dp[:,0])

plt.style.use('ggplot')
fig = plt.figure(figsize=(8, 8))

plt.plot(np_fpr, np_tpr, label = 'Non private model')
plt.plot(dp_fpr, dp_tpr, label = 'Differentially private model')

plt.legend(facecolor = 'w')

plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.show()
```
![Image](https://i.imgur.com/Mo05HuV.png)

---
## Conclusion
Computer vision stands at the forefront of artificial intelligence, empowering machines to interpret and understand visual information akin to human perception. The integration of image processing techniques is crucial for enhancing the capabilities of AI systems, enabling them to perform tasks such as object recognition, image enhancement, and automated editing with remarkable efficiency.

The diverse applications of image processing—ranging from improving image quality to facilitating creative expression—underscore its significance in various fields, including healthcare, where it plays a vital role in medical image analysis. The case study on pneumonia detection illustrates the dual challenge of achieving high accuracy in medical diagnostics while safeguarding patient privacy. By incorporating Differential Privacy into deep learning models, we can ensure that sensitive data remains protected, thus addressing a critical concern in the utilization of AI in healthcare.

As we continue to explore the potential of computer vision and image processing, it is evident that these technologies will not only enhance our ability to analyze and interpret visual data but also pave the way for innovative solutions that prioritize both effectiveness and ethical considerations.

---
## Presentation Development
![Presentation Title Page](https://i.imgur.com/BkCO1ua.png)
### Slide 1: Introduction to Computer Vision and Image Processing
![Slide 1: Introduction to Computer Vision and Image Processing](https://i.imgur.com/leqQMoX.png)
### Slide 2: Types of Image Processing Techniques
![Slide 2: Types of Image Processing Techniques](https://i.imgur.com/vuQaYJP.png)
### Slide 3: Case Study Overview
![Slide 3: Case Study Overview](https://i.imgur.com/pK5wdkY.png)
### Slide 4: Image Processing Implementation
![Slide 4: Image Processing Implementaion](https://i.imgur.com/qTuhj7v.png)
### Slide 5: Conclusion
![Slide 5: Conclusion](https://i.imgur.com/8ELbntv.png)

---
## Extension Activities
### Research an Emerging Form of Image Processing
Recent advancements in image processing are significantly shaping the landscape of artificial intelligence (AI) systems, particularly through the integration of deep learning techniques such as [Generative Adversarial Networks (GANs)](https://en.wikipedia.org/wiki/Generative_adversarial_network) and [Convolutional Neural Networks (CNNs)](https://en.wikipedia.org/wiki/Convolutional_neural_network). These emerging techniques enhance the ability of AI to analyze and interpret visual data, which has profound implications for various applications.
#### Emerging Techniques in Image Processing
##### Generative Adversarial Networks (GANs)
GANs are a notable advancement in image processing that have gained traction since their introduction by Ian Goodfellow in 2014. They consist of two neural networks—the generator and the discriminator—that compete against each other. The generator creates images, while the discriminator evaluates them against real images. This adversarial process leads to the generation of high-quality synthetic images, which can be used in various applications, from art creation to enhancing training datasets for AI models. [^12]
##### Convolutional Neural Networks (CNNs)
CNNs have revolutionized image processing by allowing for efficient image recognition and classification. Unlike traditional methods that process images pixel by pixel, CNNs analyze images in patches, enabling them to capture spatial hierarchies and patterns more effectively. This capability is crucial for applications such as facial recognition, medical imaging, and autonomous vehicles, where understanding complex visual information is essential. [^13]
##### Image Restoration and Enhancement Techniques
Techniques such as image restoration and enhancement are also evolving. These methods aim to improve the quality of images by removing noise and correcting distortions, which is particularly important in medical imaging and remote sensing. Advanced algorithms are now capable of restoring old or damaged images, thereby preserving historical data and improving diagnostic accuracy in healthcare. [^14]
#### Potential Impact on Future AI Systems
##### Enhanced Visual Understanding
The integration of these advanced image processing techniques into AI systems will lead to improved visual understanding. AI will be able to interpret and analyze images with greater accuracy, enabling applications in various fields, including healthcare, security, and entertainment. For example, in healthcare, enhanced imaging techniques can lead to better diagnostics and treatment planning by providing clearer images of medical conditions. [^15]
##### Real-Time Processing and Automation
As image processing techniques become more efficient, AI systems will be able to process visual data in real-time. This capability is crucial for applications such as autonomous vehicles, where immediate analysis of the surrounding environment is necessary for safe navigation. The trend towards more powerful AI accelerators will further enhance this capability, making real-time image processing more accessible and cost-effective. [^16]
##### Broader Applications in IoT and Smart Systems
The convergence of image processing with the Internet of Things (IoT) is expected to create new opportunities for smart systems. Enhanced image processing capabilities will allow IoT devices to monitor environments more effectively, leading to improved safety and efficiency in sectors like manufacturing, agriculture, and urban planning. For instance, smart surveillance systems can leverage advanced imaging techniques to provide real-time analytics and alerts. [^16]
#### Ethical Considerations and Challenges
While the advancements in image processing present numerous opportunities, they also raise ethical considerations, particularly regarding privacy and security. The ability to generate realistic images can lead to misuse, such as deepfakes, which pose significant challenges for trust and authenticity in digital media. [^16]

![Slide 6: Emerging Form of Image Processing](https://i.imgur.com/UgvC7Aa.png)

---
## References
[^1]: SAS Institute Inc. (2024). _Computer Vision: What it is and why it matters?_ https://www.sas.com/en_th/insights/analytics/computer-vision.html
[^2]: Alooba. (2024). _Visual Data Processing: Everything You Need to Know When Assessing Visual Data Processing Skills_. https://www.alooba.com/skills/concepts/artificial-intelligence/visual-data-processing/
[^3]: Alvi, F. (2023). _Deep Learning For Computer Vision: Essential Models and Practical Real-World Applications_. OpenCV. https://opencv.org/blog/deep-learning-with-computer-vision/
[^4]: Vision Platform. (2024). _Understanding Image Recognition: Algorithms, Machine Learning, and Uses_. Studio Slash. https://visionplatform.ai/image-recognition/
[^5]: Amazon. (2024). _What is Facial Recognition?_ Amazon Web Services, Inc. https://aws.amazon.com/what-is/facial-recognition/
[^6]: Algoscale Technologies. (2022). _Anomaly Detection in Computer Vision_. https://algoscale.com/blog/anomaly-detection-in-computer-vision/
[^7]: Damen, A. (2023). _How AI is transforming the way we edit and enhance photos?_ Photoroom. https://www.photoroom.com/blog/ai-in-photo-editing
[^8]: Snell, A. (2024). _Revolutionizing Video Editing with AI: The Future of Automation - Upbeat Geek_. Upbeat Geek. https://www.upbeatgeek.com/revolutionizing-video-editing-with-ai-the-future-of-automation/
[^9]: Aarthy, R. (2023). _Filtering in Image Processing_. Scaler Topics. https://www.scaler.com/topics/filtering-in-image-processing/
[^10]: Contributing Writer. (2024). _Edge Detection in Image Processing: An Introduction_. Roboflow Blog. https://blog.roboflow.com/edge-detection/
[^11]: International Business Machines Corporation. (2024). _What Is Image Segmentation?_ IBM. https://www.ibm.com/topics/image-segmentation
[^12]: Brownlee, J. (2019). _A Gentle Introduction to Generative Adversarial Networks (GANs)_. Machine Learning Mastery. https://machinelearningmastery.com/what-are-generative-adversarial-networks-gans/
[^13]: Datacamp. (2023). _An Introduction to Convolutional Neural Networks (CNNs)_. Data Camp. https://www.datacamp.com/tutorial/introduction-to-convolutional-neural-networks-cnns
[^14]: Wali, A., Naseer, A., Tamoor, M., & Gilani, S. (2023). _Recent progress in digital image restoration techniques: A review_. Digital Signal Processing. https://www.sciencedirect.com/science/article/abs/pii/S1051200423002828
[^15]: Kundu, R. (2024). _Image Processing: Techniques, Types, & Applications_. V7. https://www.v7labs.com/blog/image-processing-guide
[^16]: Deligiannidis, L., & Arabnia, H. (2015). _Emerging Trends in Image Processing, Computer Vision and Pattern Recognition_. Science Direct. https://www.sciencedirect.com/book/9780128020456/emerging-trends-in-image-processing-computer-vision-and-pattern-recognition
