import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
from model import ResNet  # The model
from config import resize_x, resize_y # Import resize dimensions

def classify_animals(image_paths):
    """
    Predicts the animal class for a list of image paths.

    Args:
        image_paths (list): A list of paths to the images to classify.
        device (torch.device): The device to perform computation on (CPU or GPU).

    Returns:
        list: A list of predicted class names for each image.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes).to(device)  # Instantiate the model
    net.load_state_dict(torch.load('/checkpoints/final_weights.pth',map_location=device)) # Load the trained weights
    net.eval() # Set the model to evaluation mode

    transform = transforms.Compose([
        transforms.Resize(size=(resize_x, resize_y)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    images = []
    for img_path in image_paths:
        try:
            img = Image.open(img_path).convert('RGB')  # Open and convert to RGB
            img_tensor = transform(img).unsqueeze(0).to(device) # Apply transformations and add batch dimension
            images.append(img_tensor)
        except FileNotFoundError:
            print(f"Error: Image not found at {img_path}")
            return ["Error: Image not found"] * len(image_paths)  # Or handle the error as you see fit

    if not images:
        return ["Error: No valid images provided"]

    image_batch = torch.cat(images, dim=0) # Concatenate the images into a batch

    with torch.no_grad():
        outputs = net(image_batch)
        _, predicted = torch.max(outputs, 1) # Get the predicted class indices

    class_names = ['antelope', 'badger', 'bat', 'bear', 'bee', 'beetle', 'bison', 'boar', 'butterfly', 'cat', 'caterpillar', 'chimpanzee', 'cockroach', 'cow', 'coyote', 
                   'crab', 'crow', 'deer', 'dog', 'dolphin', 'donkey', 'dragonfly', 'duck', 'eagle', 'elephant', 'flamingo', 'fly', 'fox', 'goat', 'goldfish', 'goose', 
                   'gorilla', 'grasshopper', 'hamster', 'hare', 'hedgehog', 'hippopotamus', 'hornbill', 'horse', 'hummingbird', 'hyena', 'jellyfish', 'kangaroo', 'koala',
                   'ladybugs', 'leopard', 'lion', 'lizard', 'lobster', 'mosquito', 'moth', 'mouse', 'octopus', 'okapi', 'orangutan', 'otter', 'owl', 'ox', 'oyster', 'panda', 
                   'parrot', 'pelecaniformes', 'penguin', 'pig', 'pigeon', 'porcupine', 'possum', 'raccoon', 'rat', 'reindeer', 'rhinoceros', 'sandpiper', 'seahorse', 'seal',
                   'shark', 'sheep', 'snake', 'sparrow', 'squid', 'squirrel', 'starfish', 'swan', 'tiger', 'turkey', 'turtle', 'whale', 'wolf', 'wombat', 'woodpecker', 'zebra'] 
    predicted_class_names = [class_names[i] for i in predicted]

    return predicted_class_names

