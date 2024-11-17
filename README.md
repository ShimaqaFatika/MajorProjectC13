### CSRNet-keras
Implementation of the CSRNet paper (CVPR 18) in keras-tensorflow. First ever to be done in keras-tf as of 21/9/18.

#### CVPR 2018 Paper : https://arxiv.org/abs/1802.10062
#### Our Research Paper: https://ijsiet.org/issue2024vol6.html
#### Official Pytorch Implementation : https://github.com/leeyeehoo/CSRNet-pytorch

### Dataset :
The dataset used is ShanghaiTech dataset available here : [Drive Link](https://drive.google.com/file/d/16dhJn7k4FWVwByRsQAEpl9lwjuV03jVI/view)

The dataset is divided into two parts, A and B. Part A consists of images with a high density of crowd. Part B consists of images with images of sparse crowd scenes.   

### Abstract

Crowd counting is known to be act of counting the total crowd present in a certain area. The people in a certain area are called a crowd. The most direct method is to actually count each person in the crowd. For example, turnstiles are often used to precisely count the number of people entering an event.

Manual counting, the traditional method employed in such scenarios, is not only time- consuming but also prone to inaccuracies. Recognising the need for an efficient and accurate solution, we propose the implementation of CSRNet, a Congested Scene Recognition Network, for crowd counting and density estimation.

In our we approach used a modified convolutional neural network equipped with a density map to tackle crowd counting effectively. Despite common hurdles such as partial obstruction and overlapping individuals, our model remains robust. The CSRNet Model processes input images to generate density maps, enabling accurate people counting. Training on the ShanghaiTech Dataset ensures the model's reliability and performance. Our project serves to streamline crowd estimation processes, offering a solution applicable across various scenarios.

### EXISTING SYSTEMS

Existing methods such as MCNN, Switch-CNN, and Contextual Pyramid CNN have been used for crowd counting. While each had its strengths, they also presented limitations. For example, Switch- CNN utilized different CNNs for different parts of a crowd scene, but managing multiple CNN regressors with different architectures added complexity to the model, leading to potential generalization issues. Contextual Pyramid CNN aimed to incorporate global and local contextual information to generate high-quality crowd density maps, but its increased complexity posed a higher risk of overfitting, especially with smaller datasets or inadequate regularization techniques. MCNN attempted to map images to crowd density maps but struggled with generating high-quality density maps.

### PROPOSED SYSTEM

CSRNet is a technique we implemented in our thesis, it deploys a deeper CNN for capturing high- level features and generating high-quality density-maps without increasing the complexity of network.
The selection of CSRNet as the preferred technique for predicting the total number of people in an image is justified due to its ability to address several shortcomings observed in other approaches.
CSRNet offers a unique approach that focuses on maximizing feature extraction from the given image while efficiently generating density maps. By leveraging the strengths of convolutional neural networks, CSRNet streamlines the process by processing the entire image to produce accurate counts of people. This approach eliminates the need for complex structures like multi- column architectures and density level classifiers, thereby reducing training time and mitigating unwanted values in density maps.
CSRNet stands out as a superior choice for predicting the count of people in images due to its ability to overcome the limitations observed in other techniques. By prioritizing feature extraction and simplifying the network architecture, CSRNet offers a more efficient and accurate solution for crowd counting tasks.

### CSRNet Architecture

CSRNet uses CNN as backbone, in our project we will be using VGG16 (Visual Geometry Group 16) . It is nothing but convolutional neural network which is 16 layers deep. We will be using this for feature extraction. The output from this is 1/8th of the given input size.
Within this we will have convolutional layers with filters of different sizes followed by max pooling layers. The sizes of filters used are 64,128,256,512, the kernel size is 3 and stride is 1. The aim of this is to extract the feature of images from different perspectives. We use max pooling after every 3 layers( a block) to reduce the spatial dimensions of the feature maps, allowing the network to focus on the most important information and discard less relevant details.
In addition to this, we will be using dialated convolutional layers of size 512. Dilated convolutional layers enable neural networks to see the bigger picture, making them better at understanding the overall context in images. The backend consists of four different configurations, each with its own set of layers and parameters. The concept behind this convolution layer is it increases the kernel size without increasing the parameters so that we can extract the low-quality features very easily.

### VGG16:

In our project, we have opted to utilize the VGG16 architecture, a 16-layer deep Convolutional Neural Network (CNN), as the backbone for feature extraction. The VGG16 model is renowned for its simplicity and effectiveness. It is designed to capture diverse features from input images, enhancing the model's ability to discern intricate details within crowded scenes.
Filter Sizes and Max Pooling: Within the VGG16 front end, convolutional layers with filters of varying sizes (64, 128, 256, 512) are employed, each followed by max pooling layers. This strategy aims to extract features from different perspectives, providing the network with a comprehensive understanding of the input images.
Reduction of Spatial Dimensions: Max pooling is strategically applied after every three layers (a block), effectively reducing the spatial dimensions of the feature maps. This process optimizes the network's focus on key information, discarding less relevant details and enhancing computational efficiency.

### Dilated Convolutional Layers:
Overview of Dilated Convolutional Layers: Our model incorporates dilated convolutional layers, a crucial component for enabling the network to perceive a broader context in images. These layers facilitate a more extensive receptive field without a proportional increase in parameters, making them adept at capturing the overall context in the input data.

### Model Analysis
Mean Square Error: 31
R - Squared : 0.94
