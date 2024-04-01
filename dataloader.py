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
import random

from torch.utils.data import WeightedRandomSampler
from collections import defaultdict


class ChestXrayDataset(ImageFolder):
    def __init__(self, root_dir, transform=None):
        super(ChestXrayDataset, self).__init__(root=root_dir, transform=transform)

class ChestXrayDatasetInMemory(ImageFolder):
    def __init__(self, root_dir, aug=None, num_workers=32):
        super().__init__(root=root_dir)
        self.samples = self.samples
        self.set_aug(aug)
        # if 'train' in root_dir or 'val' in root_dir:
        #     self.samples = self.samples[:int(len(self.samples)/3)]
        self.num_workers = num_workers
        self.class_probs = None
        self.images = []
        self.labels = []
        self.aug = aug
        
        self.load_dataset_into_memory()

    def set_aug(self, aug=False, image_net=True):
        if aug:
            self.transform = A.Compose([
                # A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8)),
                A.Rotate(limit=20, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=1),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0, rotate_limit=0, p=0.5),
                A.Perspective(scale=(0.05, 0.15), keep_size=True, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.Resize(height=224, width=224),
                (A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if image_net else A.Normalize(mean=0, std=1)),
                # A.Normalize(mean=0, std=1),
                ToTensorV2()
            ]) 
            # self.transform = A.Compose([

            #     A.Rotate(limit=30, p=0.5),
    
            #     # Zoom range of [1 - 0.2, 1 + 0.2]
            #     A.RandomScale(scale_limit=0.2, p=0.5),
                
            #     # Width and height shift range of 10% of the total width or height
            #     A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0., rotate_limit=0, p=0.5),
                
            #     # Horizontal flip
            #     A.HorizontalFlip(p=0.5),

            #     A.Resize(height=200, width=200),
                
            #     # You might also want to add normalization and ToTensor conversion for PyTorch
            #     # If you're using normalization, make sure to adjust the mean and std to your dataset's parameters
            #     # A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
            #     A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            #     ToTensorV2()
            # ]) 
            # self.transform = A.Compose([
            #     A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),  # Apply CLAHE with a 50% probability
            #     A.Resize(height=224, width=224),
            #     A.HorizontalFlip(p=0.5),  # Apply horizontal flip with a 50% probability
            #     A.VerticalFlip(p=0.5),  # Apply vertical flip with a 50% probability
            #     A.Rotate(limit=10, p=0.5),  # Rotate ±10 degrees with a 50% probability
            #     A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
            #     A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            #     A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            #     ToTensorV2()
            # ])
        else:
            self.transform = A.Compose([
                A.Resize(224, 224),  # Resize images to 200x200
                # ToTensorV2(),          # Convert images to PyTorch tensors
                (A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if image_net else A.Normalize(mean=0, std=1)),
                # A.Normalize(mean=0, std=1),
                ToTensorV2()
            ])

    def load_image_transform(self, img_label_tuple):
        img_path, label = img_label_tuple
        # image = Image.open(img_path).convert('L')
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        # np.divide(image, 255, out=image, casting='unsafe')
        # image = self.transform(image=np.array(image))['image']
        return image, label

    def load_dataset_into_memory(self):
        class_freq = defaultdict(int)
        for _, label in self.samples:
            class_freq[label] += 1

        # Invert class frequencies
        class_freq_inverted = {k: max(class_freq.values()) / v for k, v in class_freq.items()}
        total_freq = sum(class_freq_inverted.values())
        class_probabilities = {label: freq / total_freq for label, freq in class_freq_inverted.items()}
        self.class_probs = list(class_probabilities.values())
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
        
        indices = list(range(len(self.images)))
        random.shuffle(indices)

        # Use the shuffled indices to reorder both lists
        self.images = [self.images[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]


    def __getitem__(self, index):
        # if self.aug:
        #     weights_for_selection = [0.5 if x == 0 else 0.5 for x in self.labels]
        #     index = random.choices(range(len(self.labels)), weights=weights_for_selection, k=1)[0]
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

  
# # Path to your dataset directory
# dataset_path = 'data/chest_xray'

# # Load the datasets
# train_dataset = ChestXrayDatasetInMemory(os.path.join(dataset_path, 'train'), aug=True)
# val_dataset = ChestXrayDatasetInMemory(os.path.join(dataset_path, 'val'), aug=False)
# test_dataset = ChestXrayDatasetInMemory(os.path.join(dataset_path, 'test'), aug=False)

# # Create the DataLoader for each dataset
# batch_size = 32  # You can adjust this according to your system's capabilities

# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
