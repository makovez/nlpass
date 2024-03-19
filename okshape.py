import torch
from torch import nn
# Define a dummy input tensor with the shape of the AlexNet input: (batch_size, channels, height, width)
# Here, batch_size = 1, channels = 3 (RGB), height = width = 224 (standard AlexNet input size)
dummy_input = torch.randn(1, 3, 200, 200)

# Define the AlexNet features module (convolutional part)
features = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(64, 192, kernel_size=5, padding=2),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Conv2d(192, 384, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(384, 256, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(256, 256, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=3, stride=2),
)

# Pass the dummy input through each layer of the features module and print the shape after each layer
shape_changes = []
for i, layer in enumerate(features):
    dummy_input = layer(dummy_input)
    shape_changes.append(f"Layer {i+1} output shape: {dummy_input.shape}")

print(shape_changes)
