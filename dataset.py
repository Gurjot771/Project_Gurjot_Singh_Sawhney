import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from config import batchsize, resize_x, resize_y # Import config variables
from config import main_dir # Import config variable

def create_image_datasets(main_dir):
    """
    Creates the training, validation, and test datasets using ImageFolder.

    Args:
        main_dir (str): The main directory containing the train, validation, and test subdirectories.
        resize_x (int): The horizontal size to resize images to.
        resize_y (int): The vertical size to resize images to.

    Returns:
        tuple: A tuple containing the training, validation, and test datasets, and a list of class names.
            (train_dataset, val_dataset, test_dataset, class_names)
    """
    train_dir = os.path.join(main_dir, "train")
    val_dir = os.path.join(main_dir, "validation")
    test_dir = os.path.join(main_dir, "test")

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=(resize_x, resize_y)),  # config variables
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=(resize_x, resize_y)),  # config variables
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=(resize_x, resize_y)),  # config variables
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform_train)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=transform_val)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform_test)

    return train_dataset, val_dataset, test_dataset, train_dataset.classes  # Returns class names



def create_data_loaders(train_dataset, val_dataset, test_dataset):
    """
    Creates the data loaders for the training, validation, and test datasets.

    Args:
        train_dataset (Dataset): The training dataset.
        val_dataset (Dataset): The validation dataset.
        test_dataset (Dataset): The test dataset.
        batchsize (int): The batch size for the data loaders.

    Returns:
        tuple: A tuple containing the training, validation, and test data loaders.
            (train_loader, val_loader, test_loader)
    """
    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)  # config variable
    val_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False)  # config variable
    test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)

    return train_loader, val_loader, test_loader
