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

This table compares the performance of ResNet-18 and VGG models across various classes.  Metrics include Precision, Recall, and F1-Score.

| ResNet-18 Class | ResNet-18 Precision | ResNet-18 Recall | ResNet-18 F1-Score | VGG Precision | VGG Recall | VGG F1-Score |
|-----------------|--------------------|-------------------|--------------------|---------------|------------|--------------|
| antelope | 0.6852 | 0.5606 | 0.6167 | 1.0000 | 0.2045 | 0.3396 |
| badger | 0.6899 | 0.6742 | 0.6820 | 0.5701 | 0.4621 | 0.5105 |
| bat | 0.4000 | 0.4545 | 0.4255 | 0.5714 | 0.4242 | 0.4870 |
| bear | 0.4937 | 0.5909 | 0.5379 | 0.4435 | 0.3864 | 0.4130 |
| bee | 0.5127 | 0.7652 | 0.6140 | 0.8906 | 0.4318 | 0.5816 |
| beetle | 0.5301 | 0.3333 | 0.4093 | 0.4444 | 0.3030 | 0.3604 |
| bison | 0.6940 | 0.7045 | 0.6992 | 0.3988 | 0.5076 | 0.4467 |
| boar | 0.4037 | 0.4924 | 0.4437 | 0.3333 | 0.3788 | 0.3546 |
| butterfly | 0.7447 | 0.5303 | 0.6195 | 0.5893 | 0.5000 | 0.5410 |
| cat | 0.7324 | 0.7879 | 0.7591 | 0.6935 | 0.6515 | 0.6719 |
| caterpillar | 0.5185 | 0.2121 | 0.3011 | 0.2907 | 0.1894 | 0.2294 |
| chimpanzee | 0.6480 | 0.6136 | 0.6304 | 0.2872 | 0.8333 | 0.4272 |
| cockroach | 0.6949 | 0.6212 | 0.6560 | 0.5862 | 0.5152 | 0.5484 |
| cow | 0.1880 | 0.1667 | 0.1767 | 0.1040 | 0.0985 | 0.1012 |
| coyote | 0.2212 | 0.1742 | 0.1949 | 0.1402 | 0.4015 | 0.2078 |
| crab | 0.5053 | 0.7197 | 0.5938 | 0.6864 | 0.6136 | 0.6480 |
| crow | 0.5556 | 0.6439 | 0.5965 | 0.5584 | 0.6515 | 0.6014 |
| deer | 0.5299 | 0.5379 | 0.5338 | 0.5000 | 0.5379 | 0.5182 |
| dog | 0.4966 | 0.5530 | 0.5233 | 0.6400 | 0.2424 | 0.3516 |
| dolphin | 0.5396 | 0.5682 | 0.5535 | 0.4340 | 0.3485 | 0.3866 |
| donkey | 0.3913 | 0.4091 | 0.4000 | 0.4318 | 0.2879 | 0.3455 |
| dragonfly | 0.6066 | 0.5606 | 0.5827 | 0.8028 | 0.4318 | 0.5616 |
| duck | 0.6102 | 0.5455 | 0.5760 | 0.6667 | 0.1970 | 0.3041 |
| eagle | 0.6161 | 0.5227 | 0.5656 | 0.7263 | 0.5227 | 0.6079 |
| elephant | 0.6642 | 0.6742 | 0.6692 | 0.5324 | 0.5606 | 0.5461 |
| flamingo | 0.5676 | 0.6364 | 0.6000 | 0.6290 | 0.5909 | 0.6094 |
| fly | 0.6438 | 0.7121 | 0.6763 | 0.5238 | 0.6667 | 0.5867 |
| fox | 0.5167 | 0.4697 | 0.4921 | 0.3743 | 0.4848 | 0.4224 |
| goat | 0.3577 | 0.3712 | 0.3643 | 0.4714 | 0.2500 | 0.3267 |
| goldfish | 0.7143 | 0.7576 | 0.7353 | 0.7556 | 0.7727 | 0.7640 |
| goose | 0.6076 | 0.3636 | 0.4550 | 0.5854 | 0.1818 | 0.2775 |
| gorilla | 0.6220 | 0.7727 | 0.6892 | 0.6036 | 0.5076 | 0.5514 |
| grasshopper | 0.5223 | 0.6212 | 0.5675 | 0.4431 | 0.5606 | 0.4950 |
| hamster | 0.7305 | 0.7803 | 0.7546 | 0.6250 | 0.6439 | 0.6343 |
| hare | 0.5596 | 0.4621 | 0.5062 | 0.3455 | 0.4318 | 0.3838 |
| hedgehog | 0.6512 | 0.8485 | 0.7368 | 0.8029 | 0.8333 | 0.8178 |
| hippopotamus | 0.5339 | 0.4773 | 0.5040 | 0.4437 | 0.4773 | 0.4599 |
| hornbill | 0.4641 | 0.7348 | 0.5689 | 0.3643 | 0.7727 | 0.4951 |
| horse | 0.4757 | 0.6667 | 0.5552 | 0.3798 | 0.3712 | 0.3755 |
| hummingbird | 0.6667 | 0.7576 | 0.7092 | 0.6514 | 0.5379 | 0.5892 |
| hyena | 0.6838 | 0.6061 | 0.6426 | 0.3532 | 0.5833 | 0.4400 |
| jellyfish | 0.5677 | 0.6667 | 0.6132 | 0.6081 | 0.6818 | 0.6429 |
| kangaroo | 0.5455 | 0.4545 | 0.4959 | 0.4545 | 0.2652 | 0.3349 |
| koala | 0.7403 | 0.8636 | 0.7972 | 0.5575 | 0.7348 | 0.6340 |
| ladybugs | 0.7329 | 0.8106 | 0.7698 | 0.6961 | 0.5379 | 0.6068 |
| leopard | 0.8304 | 0.7045 | 0.7623 | 0.5573 | 0.5530 | 0.5551 |
| lion | 0.5946 | 0.8333 | 0.6940 | 0.7736 | 0.3106 | 0.4432 |
| lizard | 0.4455 | 0.3409 | 0.3863 | 0.4889 | 0.3333 | 0.3964 |
| lobster | 0.6911 | 0.6439 | 0.6667 | 0.6423 | 0.5985 | 0.6196 |
| mosquito | 0.7294 | 0.9394 | 0.8212 | 0.8143 | 0.8636 | 0.8382 |
| moth | 0.6016 | 0.5833 | 0.5923 | 0.5045 | 0.4242 | 0.4609 |
| mouse | 0.6644 | 0.7500 | 0.7046 | 0.7500 | 0.5227 | 0.6161 |
| octopus | 0.3394 | 0.2803 | 0.3071 | 0.4730 | 0.2652 | 0.3398 |
| okapi | 0.8611 | 0.7045 | 0.7750 | 0.9158 | 0.6591 | 0.7665 |
| orangutan | 0.6919 | 0.9015 | 0.7829 | 0.8072 | 0.5076 | 0.6233 |
| otter | 0.4786 | 0.4242 | 0.4498 | 0.4000 | 0.2727 | 0.3243 |
| owl | 0.6829 | 0.6364 | 0.6588 | 0.7529 | 0.4848 | 0.5899 |
| ox | 0.4604 | 0.4848 | 0.4723 | 0.2829 | 0.5530 | 0.3744 |
| oyster | 0.6642 | 0.6894 | 0.6766 | 0.7711 | 0.4848 | 0.5953 |
| panda | 0.8121 | 0.9167 | 0.8612 | 0.7375 | 0.8939 | 0.8082 |
| parrot | 0.5094 | 0.4091 | 0.4538 | 0.5000 | 0.3712 | 0.4261 |
| pelecaniformes | 0.5087 | 0.6667 | 0.5770 | 0.4752 | 0.5076 | 0.4908 |
| penguin | 0.7302 | 0.6970 | 0.7132 | 0.7282 | 0.5682 | 0.6383 |
| pig | 0.4420 | 0.4621 | 0.4519 | 0.3448 | 0.2273 | 0.2740 |
| pigeon | 0.6082 | 0.4470 | 0.5153 | 0.5500 | 0.3333 | 0.4151 |
| porcupine | 0.5621 | 0.6515 | 0.6035 | 0.3097 | 0.6288 | 0.4150 |
| possum | 0.6767 | 0.6818 | 0.6792 | 0.4175 | 0.6136 | 0.4969 |
| raccoon | 0.7556 | 0.7727 | 0.7640 | 0.8750 | 0.5303 | 0.6604 |
| rat | 0.5333 | 0.2424 | 0.3333 | 0.6863 | 0.2652 | 0.3825 |
| reindeer | 0.6098 | 0.5682 | 0.5882 | 0.7031 | 0.3409 | 0.4592 |
| rhinoceros | 0.6897 | 0.6061 | 0.6452 | 0.6489 | 0.4621 | 0.5398 |
| sandpiper | 0.7402 | 0.7121 | 0.7259 | 0.7101 | 0.7424 | 0.7259 |
| seahorse | 0.6393 | 0.5909 | 0.6142 | 0.5800 | 0.6591 | 0.6170 |
| seal | 0.6105 | 0.4394 | 0.5110 | 0.2436 | 0.2879 | 0.2639 |
| shark | 0.5556 | 0.4167 | 0.4762 | 0.3649 | 0.4091 | 0.3857 |
| sheep | 0.5079 | 0.4848 | 0.4961 | 0.3540 | 0.4318 | 0.3891 |
| snake | 0.6147 | 0.5076 | 0.5560 | 0.6579 | 0.3788 | 0.4808 |
| sparrow | 0.8365 | 0.6591 | 0.7373 | 0.6063 | 0.5833 | 0.5946 |
| squid | 0.4265 | 0.4394 | 0.4328 | 0.2727 | 0.3864 | 0.3197 |
| squirrel | 0.3478 | 0.3030 | 0.3239 | 0.2517 | 0.2879 | 0.2686 |
| starfish | 0.5217 | 0.4545 | 0.4858 | 0.4397 | 0.4697 | 0.4542 |
| swan | 0.6022 | 0.4242 | 0.4978 | 0.6912 | 0.3561 | 0.4700 |
| tiger | 0.8279 | 0.7652 | 0.7953 | 0.7479 | 0.6742 | 0.7092 |
| turkey | 0.7857 | 0.7500 | 0.7674 | 0.6777 | 0.6212 | 0.6482 |
| turtle | 0.5983 | 0.5303 | 0.5622 | 0.3820 | 0.5152 | 0.4387 |
| whale | 0.5395 | 0.6212 | 0.5775 | 0.2653 | 0.8561 | 0.4050 |
| wolf | 0.4737 | 0.3409 | 0.3965 | 0.2093 | 0.1364 | 0.1651 |
| wombat | 0.5049 | 0.7879 | 0.6154 | 0.4000 | 0.7121 | 0.5123 |
| woodpecker | 0.7605 | 0.9621 | 0.8495 | 0.8143 | 0.8636 | 0.8382 |
| zebra | 0.7949 | 0.9394 | 0.8611 | 0.6103 | 0.9015 | 0.7278 |

  
