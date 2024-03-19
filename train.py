import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
import torch
from dataloader import train_loader, val_loader, test_loader


import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50

from tqdm import tqdm

from model.ResNet import resnet34
from model.VGG16 import VGG16
from model.AlexNet import AlexNet
from torchmetrics import Accuracy, Precision, F1Score, Recall

class Trainer:
    def __init__(self, train_loader, val_loader, test_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize the model
        self.model = resnet50(pretrained=True)
        self.model.to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Metrics
        self.accuracy_metric = Accuracy(task='multiclass', num_classes=2,average='macro').to(self.device)
        self.precision_metric = Precision(task='multiclass', num_classes=2, average='macro').to(self.device)
        self.recall = Recall(task='multiclass', num_classes=2, average='macro').to(self.device)
        self.f1_metric = F1Score(task='multiclass', num_classes=2, average='macro').to(self.device)
        
    def train(self, num_epochs=10):
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            
            # Reset metrics at the start of each epoch
            self.accuracy_metric.reset()
            self.precision_metric.reset()
            self.f1_metric.reset()
            self.recall.reset()
            
            with tqdm(self.train_loader, unit="batch") as tepoch:
                for images, labels in tepoch:
                    tepoch.set_description(f"Epoch {epoch+1}/{num_epochs}")
                    
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    
                    # Backward and optimize
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    # Update metrics
                    self.accuracy_metric.update(outputs, labels)
                    self.precision_metric.update(outputs, labels)
                    self.f1_metric.update(outputs, labels)
                    self.recall.update(outputs, labels)
                    
                    running_loss += loss.item()
                    tepoch.set_postfix(loss=running_loss/len(self.train_loader))
            
            # Compute metrics at the end of each epoch
            accuracy = self.accuracy_metric.compute()
            precision = self.precision_metric.compute()
            f1 = self.f1_metric.compute()
            recall = self.recall.compute()
            
            # Print metrics at the end of each epoch
            print(f'Epoch {epoch+1}/{num_epochs} - Loss: {running_loss/len(self.train_loader):.2f}, '
                f'Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, F1 Score: {f1:.2f}, Recall: {recall:.2f}')
            
            self.test()
            
        # After training completes, evaluate on the test set
        self.test()

    def validate(self):
        self.model.eval()
        self.accuracy_metric.reset()
        self.precision_metric.reset()
        self.f1_metric.reset()
        self.recall.reset()
        
        with torch.no_grad(), tqdm(self.val_loader, unit="batch") as tepoch:
            for images, labels in tepoch:
                tepoch.set_description("Validating")
                
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                
                # Update metrics
                self.accuracy_metric.update(outputs, labels)
                self.precision_metric.update(outputs, labels)
                self.f1_metric.update(outputs, labels)
                self.recall.update(outputs, labels)
            
            # Compute metrics
            accuracy = self.accuracy_metric.compute()
            precision = self.precision_metric.compute()
            f1 = self.f1_metric.compute()
            recall = self.recall.compute()
            
            print(f'Validation - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, F1 Score: {f1:.2f}, Recall: {recall:.2f}')
    
    def test(self):
        self.model.eval()
        self.accuracy_metric.reset()
        self.precision_metric.reset()
        self.f1_metric.reset()
        self.recall.reset()

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                
                # Update metrics
                self.accuracy_metric.update(outputs, labels)
                self.precision_metric.update(outputs, labels)
                self.f1_metric.update(outputs, labels)
                self.recall.update(outputs, labels)

            # Compute metrics
            accuracy = self.accuracy_metric.compute()
            precision = self.precision_metric.compute()
            f1 = self.f1_metric.compute()
            recall = self.recall.compute()

            print(f'Test - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, F1 Score: {f1:.2f}, Recall: {recall:.2f}')

# Example usage assuming test_loader is defined
trainer = Trainer(train_loader, val_loader, test_loader)
trainer.train(num_epochs=20)
