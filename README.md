# Animal_Project
Classification of 90 classes of animals by a convolutional neural network 
Project Proposal-
This project aims to develop an image classification system capable of accurately identifying 90 distinct animal species from an original dataset of approximately 5400 images. 

# Dataset of Images-  
The dataset is taken from Kaggle and contains approximately 5.4K coloured images of varying size and dimensions in .jpg format. The dataset would first be modified on my local device to resize all the images to 224x224 pixel dimensions.The augmentations include- Random horizontalflip (p=0.5), Random vertical flip (p=0.05), Affine transformations (degrees=(-13,13), scale=(0.87, 1.07)), Random Resized Cropping (size=(150, 150), scale=(0.55, 0.8), ratio=(0.75, 1.66)), changing color jitter(brightness=.15, contrast=0.15, saturation=0.15, hue=0.075). Though there was a bug in my code and original images (which were supposed to be just resized to 224x224 pixel) were 150x150 pixel dimensions. Hence, in my code I have reimplemented a resize to 224x224 pixel dimensions during loading of images.

The input is 150X150 or 224x224 pixel-sized (RGB) images of animals belonging to 90 classes. The output is the name of the animal
Each image from the original dataset has been augmented 10 times. The final data set includes the original images, which are resized and scaled down to the specified dimensions. Hence, for every class there are 660 images which are divided among test, train and validation dataset.
Initially Vgg like networks were used on the model but showed lesser efficiency. Resnet18 model architecture is used in my current project.

(Source- https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals)

# Architecture: ResNet-18

ResNet-18 architecture is used for image classification. ResNet-18 uses "skip connections" to simplify the training of very deep networks.
It comprises of the following components:

BasicBlock:
The fundamental building block, containing two convolutional layers with batch normalization (conv1, bn1, conv2, bn2).
Shortcut Connection: Input is added to the output to mitigate vanishing gradients.
If input/output dimensions are the same, the input is directly passed.
If dimensions differ, a 1x1 convolution adjusts them.
Forward Pass: Input goes through convolutions, batch norms, and ReLU activations. The output of the second batch norm is added to the (possibly adjusted) original input, followed by a final ReLU.

ResNet Class:
Assembles BasicBlock modules.
Initial Layers: Initial convolution (conv1), batch norm (bn1), and max-pooling (maxpool) extract and downsample image features.
Layers: Four layers (layer1 to layer4) of stacked BasicBlock modules, created by the _make_layer method.
_make_layer Method: Creates a layer of BasicBlock modules, handling downsampling via stride in the first block of a layer.
Final Layers: Average pooling (avgpool) reduces feature map size, and a fully connected layer (fc) produces the classification output.
Forward Pass: Input goes through initial layers, the four BasicBlock layers, average pooling, and the fully connected layer.

ResNet18 Function:
Creates a ResNet-18 model with the layer configuration [2, 2, 2, 2].
Takes the number of classes as input.

Though this function is being created but in the rest of the code ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes) is being used instead of calling the function

# Other Parameters
Optimiser: AdamsW to implement weight decay to punish the network from overfitting. 
Loss Fxn: Cross-Entropy
Test: Train ratio would be approximately 80:20 
The Train Dataset is further divided into 85:15 Train and Validation image datasets. 
The final division per class becomes- 449(train), 79(validation), 132(test)

Early stoppage function has been applied during the training- 
It has been defined during the training that if average validation loss does not decrease for X(set to 8) epochs then the training stops to provide best training weights. 
# Evaluation Metric
Evaluation Metric for the network: Accuracy was the primary measure for the evaluation. Recall and precision(hence the F1 score) were also seen along with the accuracy in every epoch. In my database, there is no significant class imbalance and F1 score is not an important factor with that respect. Though, F1 score did provide meaningful insights where network was not performing equally well on all of the classes. 

# Performance of the network: 

![Training Loss, Validation Loss and Val F1 (Macro)](https://github.com/user-attachments/assets/177d0302-524f-495e-a31c-ea7bef411dbb)

The overall accuracy is around 60% on test set and the F1 score have been disclosed below for all classes-
A comparison with a VGG Network having many layers is given for the same which gave 47 percent accuracy. The architecture of Vgg network is also provided after this section. 
![Training Loss, Validation Loss and F1 Score](https://github.com/user-attachments/assets/f5a60929-7f33-4064-9aa8-da6b57ac94c7)

<span style="background-color: #ffb3b3;">This table compares the performance of ResNet-18 and VGG models across various classes.  Metrics include Precision, Recall, and F1-Score. The classes where Vgg network is performing better are made bold. In the rest cases, the Vgg network performs very similar or worse then resnet network.

