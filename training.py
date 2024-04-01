from train import Trainer
import torch
import os
from dataloader import ChestXrayDatasetInMemory, DataLoader
from model.ResNet import resnet34
from model.VGG16 import VGG16
from model.AlexNet import AlexNet
from train import Trainer
import numpy as np
import torch.nn as nn
import random 


def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # You can choose any seed number you like

import torchvision.models as models
from torchvision.models import AlexNet_Weights
dataset_path = 'data/chest_xray'
batch_size = 32
# model = models.alexnet(weights=AlexNet_Weights.DEFAULT)
# num_ftrs = model.classifier[6].in_features  # Get the number of features in input to the final fully connected layer
# model.classifier[6] = nn.Linear(num_ftrs, 2)
model = AlexNet()
train_dataset = ChestXrayDatasetInMemory(os.path.join(dataset_path, 'train'), aug=True)
test_dataset = ChestXrayDatasetInMemory(os.path.join(dataset_path, 'test'), aug=False)

train_dataset.set_aug(aug=True, image_net=False)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the Trainer
trainer = Trainer(train_loader, test_loader, model=model, weights=None, lr=1e-4)

train_metrics = trainer.train(num_epochs=10)  # Ensure your train method returns the validation metric of interest
test_metrics = trainer.test()  # Ensure your train method returns the validation metric of interest