# Image-Denoising Using Deep Learning Algorithms

Link to the documentation-  https://drive.google.com/file/d/1B9lrtgPX8yNt7qoZhTtlhYvf-U4WXE24/view

The goal of this project is to develop an image denoising algorithm utilizing deep learning techniques, with a particular emphasis on convolutional neural networks (CNNs). Image denoising is an essential preprocessing step in numerous image processing applications, intended to eliminate noise from images while maintaining crucial details


Libraries used: Numpy, matplotlib, tensorflow, keras

# Project Overview:

In this project, we utilized a convolutional neural network (CNN) architecture, widely favored in image processing tasks for its proficiency in capturing spatial hierarchies within images. Our CNN model comprises several convolutional layers, each succeeded by activation and pooling layers, which collectively facilitate the extraction and consolidation of image features.
Convolutional Layers: These layers extract distinctive features from the input images.
LeakyReLU Activation: Implemented to maintain a small, non-zero gradient for negative inputs, thereby mitigating the "dying ReLU" issue and fostering improved gradient flow.
Pooling Layers: Employed to reduce the spatial dimensions of feature maps, thereby decreasing computational complexity.
Upsampling Layers: These layers are utilized to restore the original spatial dimensions of feature maps, ensuring compatibility with the original image size.

We partitioned the data into training and testing sets, allocating 80% for training and 20% for testing. Subsequently, we created train_ds and test_ds datasets for our custom-built model, utilizing a batch size of 32 during training.

# Basic Components of the model

The model consists of 5 convolutional layers, 2 max pooling layers, and 2 sampling layers, which are essential components of the architecture.
This setup is designed to remove noise from low-quality noisy images, converting them into higher quality images that closely resemble the
original clean, noise-free versions.

# Results and Findings:

We evaluated the loss using both the binary cross-entropy and mean squared error (MSE) functions while adjusting the number of epochs and batch size. The binary cross-entropy consistently outperformed MSE, achieving lower loss values. Specifically, optimal results were obtained with 100 epochs, a batch size of 32, and an 80:20 train-to-test ratio.

# PSNR Calculations

Binary cross-entropy computes the loss between the clean and denoised images. Peak Signal-to-Noise Ratio (PSNR) is subsequently derived from this loss. PSNR serves as a metric to gauge the fidelity of the denoised image compared to the original clean image. Higher PSNR values signify superior image quality with reduced noise in the denoised output.

The PSNR value obtained for the denoised images using our CNN model is 
For the Test Data : 17.737285614013672 dB.
For the Train Data : 18.038999557495117 dB. 
This indicates a significant reduction in noise and a good preservation of image details.

# Scope for Improvements

Data Augmentation: Enhance the diversity of training data by introducing additional variations such as rotation, scaling, and flipping. This approach aims to bolster the model's resilience and generalization capabilities.
Advanced Architectures: Explore sophisticated models like U-Net, DnCNN, or architectures based on Generative Adversarial Networks (GANs) to potentially achieve superior denoising performance.
Hyperparameter Tuning: Fine-tune key parameters such as learning rates, batch sizes, and epochs to maximize the model's effectiveness and overall performance.






