import os
import shutil
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from PIL import Image
from hist_eq import *
import cv2
import matplotlib.pyplot as plt

class ImageDatasetFolder(Dataset):
        
    def __init__(self, data_dir, data_obj):
        # Data augmentation and normalization for training
        # Just normalization for validation
        self.data_obj = data_obj
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        
        self.data_dir = data_dir
        self.image_dataset = datasets.ImageFolder(
            root = os.path.join(self.data_dir, self.data_obj),
            transform = self.data_transforms[self.data_obj])
    
       
    def __getitem__(self, idx):
        image, label = self.image_dataset[idx]
        return image, label
    
    def __len__(self):
        return len(self.image_dataset)
    
    def num_classes(self):
        return len(self.image_dataset.classes)


class ImageDataset(Dataset):
        
    def __init__(self, image_paths, image_labels, class_to_idx, data_obj):
        # Data augmentation and normalization for training
        # Just normalization for validation
        self.data_obj = data_obj
        self.class_to_idx = class_to_idx
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        
        self.image_paths = image_paths.values
        self.label_values = torch.as_tensor(image_labels.values)
        self.classes = list(self.class_to_idx.keys())
        
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        image=np.array(image)
        hist= ahe(image)
        image=Image.fromarray(hist)
        image=image.convert('RGB')
        '''hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split((hsv))
        hist_v = hist_equalization(v)
        # Merge back the channel
        merged_hist = cv2.merge((h, s, hist_v))
        hist = cv2.cvtColor(merged_hist, cv2.COLOR_HSV2BGR)

        ahe_v = ahe(v)
        merged_ahe = cv2.merge((h, s, ahe_v))
        ahe_img = cv2.cvtColor(merged_ahe, cv2.COLOR_HSV2BGR)
        pil_img=Image.fromarray(ahe_img)
        rgb_img = pil_img.convert('RGB')'''
        image = self.data_transforms[self.data_obj](image)
        label = self.label_values[idx].item()
        
        return image, label
    
    def __len__(self):
        return len(self.image_paths)
    
    def num_classes(self):
        return len(self.classes)


def format_file_path(file_path):
    # Get the image filename
    
    
    # Replace "%5C" characters with "/"
    sub_path = file_path.replace('\\', '/')
    
    return sub_path


def prepare_datafile(csv_file):
    """
        This Function will prepare the image csv file for the training, validation,
        and testing loops for the Image Classifier.

        It accepts the csv file exported from Label Studio, cleans the file path
        and returns a dataframe with the file path and file label.
    """
    # Read the CSV file
    df = pd.read_csv(csv_file, usecols=['image_path', 'label'])

    # Formats the images filepaths
    df['image_path'] = [format_file_path(i) for i in df['image_path']]
    
    return df

def get_images_from_source(source_folder, destination_folder, num_images=500):
    # Get a list of all image files in the source folder
    image_files = [os.path.join(source_folder, f) for f in os.listdir(source_folder) if f.endswith('.jpg') or f.endswith('.png')]

    # Shuffle the list of image files
    random.shuffle(image_files)

    # Move the first num_images images to the destination folder
    for i in range(num_images):
        image_file = image_files[i]
        shutil.move(image_file, os.path.join(destination_folder, os.path.basename(image_file)))

    print(f"Moved {num_images} images from {source_folder} to {destination_folder}")

def create_img_label_dict(image_folder, mask_folder):
    image_label_dict = {}
    for mask_filename in os.listdir(mask_folder):
    # Initialize an empty dictionary to store image path: label pairs
        
    # Extract the image name from the mask filename
        image_name = mask_filename.split("_label")[0]
    
    # Get the full path to the mask file
        mask_path = os.path.join(mask_folder, mask_filename)
    
    # Determine the label based on the mask filename
        label = mask_filename.split("_")[-1].split(".")[0]  # Extract label from mask filename
        label=int(label[-1])
    # Construct the full path to the corresponding image
        image_path = os.path.join(image_folder, image_name)+'.png'
    
    # Add the image path and label to the dictionary
        image_label_dict[image_path] = label
    return image_label_dict