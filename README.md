# Animal_Project
Classification of 90 classes of animals by a convolutional neural network 
Project Proposal-
This project aims to develop an image classification system capable of accurately identifying 90 distinct animal species from an original dataset of approximately 5400 images. 

# Dataset of Images-  
The dataset is taken from Kaggle and contains approximately 5.4K coloured images of varying size and dimensions in .jpg format. The dataset would first be modified on my local device to resize and augment all the images to 224x224 pixel dimensions.The augmentations include- Random horizontalflip (p=0.5), Random vertical flip (p=0.05), Affine transformations (degrees=(-13,13), scale=(0.87, 1.07)), Random Resized Cropping (size=(150, 150), scale=(0.55, 0.8), ratio=(0.75, 1.66)), changing color jitter(brightness=.15, contrast=0.15, saturation=0.15, hue=0.075). Though there was a bug in my code and original images (which were supposed to be just resized to 224x224 pixel and added along with augmented images) were 150x150 pixel dimensions. Hence in my code in the training, I have reimplemented a resize to 224x224 pixel dimensions during loading of images.

The input is 150X150 or 224x224 pixel-sized (RGB) images of animals belonging to 90 classes. The output is the name of the animal
Each image from the original dataset has been augmented 10 times. The final data set includes the original images( which are resized and scaled down to the specified dimensions) along with augmented images. Hence, for every class, there are 660 images which are divided among test, train and validation dataset.
Initially Vgg like networks were used on the model but showed lesser efficiency. The accuracy ranged from 27% to 47% on test dataset. Resnet18 model architecture is used in my current project.

(Source- https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals)

(Source for the Zip file for dataset on which the model is trained- https://drive.google.com/drive/folders/1M6wfgb9GClirqBbNUiKIEfRPIZWsEOEK?usp=drive_link)

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

Though this function is being created but in the rest of the code ResNet(BasicBlock, [2, 2, 2, 2], num_classes) is being used instead of calling the function

# Other Parameters
Optimiser: AdamsW to implement weight decay to punish the network from overfitting. 
Loss Fxn: Cross-Entropy
Test: Train ratio would be approximately 80:20 
The Train Dataset is further divided into 85:15 Train and Validation image datasets. 
The final division per class becomes- 449(train), 79(validation), 132(test)

Early stoppage function has been applied during the training- 
It has been defined during the training that if average validation loss does not decrease for X(set to 8) epochs then the training stops to provide best training weights. 
# Evaluation Metric
Evaluation Metric for the network: Accuracy was the primary measure for the evaluation. Recall and precision(hence the F1 score) were also seen along with the accuracy in every epoch on validation dataset . In my database, there is no significant class imbalance and F1 score is not an important factor with that respect. Though, F1 score did provide meaningful insights where network was not performing equally well on all of the classes. 

# Performance of the network: 

![Training Loss, Validation Loss and Val F1 (Macro)](https://github.com/user-attachments/assets/177d0302-524f-495e-a31c-ea7bef411dbb)

The overall accuracy is around 60% on test set and the F1 score have been disclosed below for all classes on test dataset-
| Animal        |   precision |   recall |   f1-score |   support |
|:--------------|------------:|---------:|-----------:|----------:|
| antelope      |    0.6852  |   0.5606 |     0.6167 |       132 |
| badger        |    0.6899  |   0.6742 |     0.682  |       132 |
| bat           |    0.4     |   0.4545 |     0.4255 |       132 |
| bear          |    0.4937  |   0.5909 |     0.5379 |       132 |
| bee           |    0.5127  |   0.7652 |     0.614  |       132 |
| beetle        |    0.5301  |   0.3333 |     0.4093 |       132 |
| bison         |    0.694   |   0.7045 |     0.6992 |       132 |
| boar          |    0.4037  |   0.4924 |     0.4437 |       132 |
| butterfly     |    0.7447  |   0.5303 |     0.6195 |       132 |
| cat           |    0.7324  |   0.7879 |     0.7591 |       132 |
| caterpillar   |    0.5185  |   0.2121 |     0.3011 |       132 |
| chimpanzee    |    0.648   |   0.6136 |     0.6304 |       132 |
| cockroach     |    0.6949  |   0.6212 |     0.656  |       132 |
| cow           |    0.188   |   0.1667 |     0.1767 |       132 |
| coyote        |    0.2212  |   0.1742 |     0.1949 |       132 |
| crab          |    0.5053  |   0.7197 |     0.5938 |       132 |
| crow          |    0.5556  |   0.6439 |     0.5965 |       132 |
| deer          |    0.5299  |   0.5379 |     0.5338 |       132 |
| dog           |    0.4966  |   0.553  |     0.5233 |       132 |
| dolphin       |    0.5396  |   0.5682 |     0.5535 |       132 |
| donkey        |    0.3913  |   0.4091 |     0.4    |       132 |
| dragonfly     |    0.6066  |   0.5606 |     0.5827 |       132 |
| duck          |    0.6102  |   0.5455 |     0.576  |       132 |
| eagle         |    0.6161  |   0.5227 |     0.5656 |       132 |
| elephant      |    0.6642  |   0.6742 |     0.6692 |       132 |
| flamingo      |    0.5676  |   0.6364 |     0.6    |       132 |
| fly           |    0.6438  |   0.7121 |     0.6763 |       132 |
| fox           |    0.5167  |   0.4697 |     0.4921 |       132 |
| goat          |    0.3577  |   0.3712 |     0.3643 |       132 |
| goldfish      |    0.7143  |   0.7576 |     0.7353 |       132 |
| goose         |    0.6076  |   0.3636 |     0.455  |       132 |
| gorilla       |    0.622   |   0.7727 |     0.6892 |       132 |
| grasshopper   |    0.5223  |   0.6212 |     0.5675 |       132 |
| hamster       |    0.7305  |   0.7803 |     0.7546 |       132 |
| hare          |    0.5596  |   0.4621 |     0.5062 |       132 |
| hedgehog      |    0.6512  |   0.8485 |     0.7368 |       132 |
| hippopotamus  |    0.5339  |   0.4773 |     0.504  |       132 |
| hornbill      |    0.4641  |   0.7348 |     0.5689 |       132 |
| horse         |    0.4757  |   0.6667 |     0.5552 |       132 |
| hummingbird    |    0.6667  |   0.7576 |     0.7092 |       132 |
| hyena         |    0.6838  |   0.6061 |     0.6426 |       132 |
| jellyfish     |    0.5677  |   0.6667 |     0.6132 |       132 |
| kangaroo      |    0.5455  |   0.4545 |     0.4959 |       132 |
| koala         |    0.7403  |   0.8636 |     0.7972 |       132 |
| ladybugs      |    0.7329  |   0.8106 |     0.7698 |       132 |
| leopard       |    0.8304  |   0.7045 |     0.7623 |       132 |
| lion          |    0.5946  |   0.8333 |     0.694  |       132 |
| lizard        |    0.4455  |   0.3409 |     0.3863 |       132 |
| lobster       |    0.6911  |   0.6439 |     0.6667 |       132 |
| mosquito      |    0.7294  |   0.9394 |     0.8212 |       132 |
| moth          |    0.6016  |   0.5833 |     0.5923 |       132 |
| mouse         |    0.6644  |   0.75   |     0.7046 |       132 |
| octopus       |    0.3394  |   0.2803 |     0.3071 |       132 |
| okapi         |    0.8611  |   0.7045 |     0.775  |       132 |
| orangutan      |    0.6919  |   0.9015 |     0.7829 |       132 |
| otter         |    0.4786  |   0.4242 |     0.4498 |       132 |
| owl           |    0.6829  |   0.6364 |     0.6588 |       132 |
| ox            |    0.4604  |   0.4848 |     0.4723 |       132 |
| oyster        |    0.6642  |   0.6894 |     0.6766 |       132 |
| panda         |    0.8121  |   0.9167 |     0.8612 |       132 |
| parrot        |    0.5094  |   0.4091 |     0.4538 |       132 |
| pelecaniformes |    0.5087  |   0.6667 |     0.577  |       132 |
| penguin       |    0.7302  |   0.697  |     0.7132 |       132 |
| pig           |    0.442   |   0.4621 |     0.4519 |       132 |
| pigeon        |    0.6082  |   0.447  |     0.5153 |       132 |
| porcupine     |    0.5621  |   0.6515 |     0.6035 |       132 |
| possum        |    0.6767  |   0.6818 |     0.6792 |       132 |
| raccoon       |    0.7556  |   0.7727 |     0.764  |       132 |
| rat           |    0.5333  |   0.2424 |     0.3333 |       132 |
| reindeer      |    0.6098  |   0.5682 |     0.5882 |       132 |
| rhinoceros    |    0.6897  |   0.6061 |     0.6452 |       132 |
| sandpiper     |    0.7402  |   0.7121 |     0.7259 |       132 |
| seahorse      |    0.6393  |   0.5909 |     0.6142 |       132 |
| seal          |    0.6105  |   0.4394 |     0.511  |       132 |
| shark         |    0.5556  |   0.4167 |     0.4762 |       132 |
| sheep         |    0.5079  |   0.4848 |     0.4961 |       132 |
| snake         |    0.6147  |   0.5076 |     0.556  |       132 |
| sparrow       |    0.8365  |   0.6591 |     0.7373 |       132 |
| squid         |    0.4265  |   0.4394 |     0.4328 |       132 |
| squirrel      |    0.3478  |   0.303  |     0.3239 |       132 |
| starfish      |    0.5217  |   0.4545 |     0.4858 |       132 |
| swan          |    0.6022  |   0.4242 |     0.4978 |       132 |
| tiger         |    0.8279  |   0.7652 |     0.7953 |       132 |
| turkey        |    0.7857  |   0.75   |     0.7674 |       132 |
| turtle        |    0.5983  |   0.5303 |     0.5622 |       132 |
| whale         |    0.5395  |   0.6212 |     0.5775 |       132 |
| wolf          |    0.4737  |   0.3409 |     0.3965 |       132 |
| wombat        |    0.5049  |   0.7879 |     0.6154 |       132 |
| woodpecker    |    0.7605  |   0.9621 |     0.8495 |       132 |
| zebra         |    0.7949  |   0.9394 |     0.8611 |       132 |



