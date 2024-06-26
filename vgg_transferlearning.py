import time
# import numpy as np
import torch
import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from PIL import Image
from utils.data_handlers import *
from utils.data_loading import *

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

path = 'vgg_16/'

# Device
DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print('Device:', DEVICE)

# hyperparameter
random_seed = 1
learning_rate = 0.001
num_epochs = 15
batch_size = 10

num_classes = 2  # class number
IMG_SIZE = (340, 340)   # resize image
# IMG_MEAN = [0.485, 0.456, 0.406]
# IMG_STD = [0.229, 0.224, 0.225]

transforms = transforms.Compose({
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor()
#     transforms.Normalize(IMG_MEAN, IMG_STD)
})


class MyDataset(Dataset):
    def __init__(self, root, datatxt, transform=None, target_transform=None):
        super(MyDataset, self).__init__()
        with open(root + datatxt, 'r') as f:
            imgs = []
            for line in f:
                line = line.rstrip()
                words = line.split()
                imgs.append((words[0], int(words[1])))
            self.root = root
            self.imgs = imgs
            self.transform = transform
            self.target_transform = target_transform

    def __getitem__(self, index):
        f, label = self.imgs[index]
        img = Image.open(self.root + f).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

dir_train_img='data\\train_vgg\\cropped_images'
dir_train_mask='data\\train_vgg\\cropped_labels'
dir_val_img='data\\val_vgg\\cropped_images'
dir_val_mask='data\\val_vgg\\cropped_labels'
dir_test_img='data\\test_vgg\\cropped_images'
dir_test_mask='data\\test_vgg\\cropped_labels'

train_dict=create_img_label_dict(dir_train_img, dir_train_mask)
val_dict=create_img_label_dict(dir_val_img, dir_val_mask)
test_dict=create_img_label_dict(dir_train_img, dir_test_mask)

train_data = CustomDataset_VGG(train_dict)
val_data=CustomDataset_VGG(val_dict)
test_data = CustomDataset_VGG(test_dict)

train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1)


"""
initial model
"""
model = models.vgg16(weights=True)
for param in model.parameters():
    param.requires_grad = False
# model = models.vgg16(pretrained=False)
    
model.classifier[6] = nn.Linear(4096,num_classes)
model = model.to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

"""
train
"""
for epoch in range(num_epochs):
    start = time.perf_counter()
    model.train()
    running_loss = 0.0
    correct_pred = 0
    for index, data in enumerate(train_loader):
        image, label = data
        image = image.to(DEVICE)
        label = label.to(DEVICE)
        y_pred = model(image)
        
        _, pred = torch.max(y_pred, 1)
        correct_pred += (pred == label).sum()
        
        loss = criterion(y_pred, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
                
        running_loss += float(loss.item())
    end = time.perf_counter()
    print('epoch {}/{}\tTrain loss: {:.4f}\tTrain accuracy: {:.2f}%'.
          format(epoch + 1, num_epochs, running_loss / (index + 1), correct_pred.item() / (batch_size * (index + 1)) * 100))
    print('Time: {:.2f}s'.format(end - start))
print('Finished training!')

"""
test
"""
test_loss = 0.0
correct_pred = 0
for _, data in enumerate(test_loader):
    image, label = data
    image = image.to(DEVICE)
    lable = label.to(DEVICE)
    y_pred = model(image)

    _, pred = torch.max(y_pred, 1)
    correct_pred += (pred == label).sum()
    
    loss = criterion(y_pred, label)
    test_loss += float(loss.item())
print('Test loss: {:.4f}\tTest accuracy: {:.2f}%'.format(test_loss / 12, correct_pred.item() / 120 * 100))