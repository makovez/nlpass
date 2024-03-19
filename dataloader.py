import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from PIL import Image
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ChestXrayDataset(ImageFolder):
    def __init__(self, root_dir, transform=None):
        super(ChestXrayDataset, self).__init__(root=root_dir, transform=transform)

class ChestXrayDatasetInMemory(ImageFolder):
    def __init__(self, root_dir, aug=None, num_workers=32):
        
        super().__init__(root=root_dir)
        if aug:
            self.transform = A.Compose([
                A.Resize(200, 200),
                A.Normalize(mean=0, std=1),
                A.Rotate(limit=20),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=1),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0, rotate_limit=0, p=0.5),
                A.Perspective(scale=(0.05, 0.15), keep_size=True, p=0.5),
                ToTensorV2()
            ]) 
        else:
            self.transform = A.Compose([
                A.Resize(200, 200),  # Resize images to 200x200
                # ToTensorV2(),          # Convert images to PyTorch tensors
                A.Normalize(mean=0, std=1),  # Normalize images
                ToTensorV2()
            ])
        # if 'train' in root_dir or 'val' in root_dir:
        #     self.samples = self.samples[:int(len(self.samples)/3)]
        self.num_workers = num_workers
        self.images = []
        self.labels = []
        self.aug = aug
        
        self.load_dataset_into_memory()

    def load_image_transform(self, img_label_tuple):
        img_path, label = img_label_tuple
        # image = Image.open(img_path).convert('L')
        image = Image.open(img_path)
        image = np.array(image)
        # image = self.transform(image=np.array(image))['image']
        return image, label

    def load_dataset_into_memory(self):
        # Using ThreadPoolExecutor to parallelize image loading
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:  # Adjust max_workers based on your CPU
            # Prepare for submission of tasks and progress tracking
            futures = [executor.submit(self.load_image_transform, img_label) for img_label in self.samples]
            results = []
            
            # Process as tasks complete
            for future in tqdm(as_completed(futures), total=len(futures), desc="Loading dataset into memory"):
                image, label = future.result()
                self.images.append(image)
                self.labels.append(label)


    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = self.transform(image=image)['image']
        return image, label

# class ChestXrayDatasetInMemory(ImageFolder):
#     def __init__(self, root_dir, aug=None):
#         self.transform = transforms.Compose([
#             transforms.Resize((200, 200)),  # Resize images to 200x200
#             transforms.Grayscale(),
#             transforms.ToTensor(),          # Convert images to PyTorch tensors
#             transforms.Normalize(mean=0, std=1),  # Normalize images
#         ])
#         super().__init__(root=root_dir, transform=self.transform)
#         self.samples = self.samples
#         self.images = []
#         self.labels = []
#         self.aug = aug
#         self.load_dataset_into_memory()

#     def load_dataset_into_memory(self):
#         for img_path, label in tqdm(self.samples, desc="Loading dataset into memory"):
#             # Load image and apply transformations
#             image = Image.open(img_path).convert("RGB")  # Convert to RGB
#             image = self.transform(image)
#             if self.aug is not None:
#                 image = self.aug(image)
#             print(image.shape)
#             self.images.append(image)
#             self.labels.append(label)

#     def __getitem__(self, index):
#         # Return the preloaded image and label
#         return self.images[index], self.labels[index]

# Specify the transformations for preprocessing the images

  
# Path to your dataset directory
dataset_path = 'data/chest_xray'

# Load the datasets
train_dataset = ChestXrayDatasetInMemory(os.path.join(dataset_path, 'train'), aug=True)
val_dataset = ChestXrayDatasetInMemory(os.path.join(dataset_path, 'val'), aug=False)
test_dataset = ChestXrayDatasetInMemory(os.path.join(dataset_path, 'test'), aug=False)

# Create the DataLoader for each dataset
batch_size = 64  # You can adjust this according to your system's capabilities

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
