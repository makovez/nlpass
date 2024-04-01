import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch

import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from torchmetrics import Accuracy, Precision, F1Score, Recall
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights
import torch
import numpy as np
import random, timm



class CustomCrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(CustomCrossEntropyLoss, self).__init__()

    def forward(self, input, target, weights = None):
        flat_labels = target.reshape(-1)
        class_counts = torch.bincount(flat_labels)
        total_pixels = torch.sum(class_counts)
        class_frequencies = class_counts / (total_pixels + 1e-6)
        class_weights = 1.0 / (class_frequencies  + 1e-6)
        class_weights /= torch.sum(class_weights)
        log_softmax = F.log_softmax(input, dim=1)

        #weighted_log_softmax = log_softmax * weights.unsqueeze(1)
        loss = F.nll_loss(log_softmax, target, weight=class_weights, reduction='mean')
        # loss = F.nll_loss(log_softmax, target, reduction='mean')
        # edge_weights = calc_edge_weights(target.float())
        # loss = loss * edge_weights
        # loss = loss.sum() / edge_weights.sum()  # Scale loss
        #loss = tensor(nan, device='cuda:0', grad_fn=<NllLossBackward0>)
       
        return loss
class EfficientNetV2Binary(nn.Module):
    def __init__(self, num_classes=2):
        super(EfficientNetV2Binary, self).__init__()
        self.efficientnetv2 = timm.create_model('efficientnetv2_rw_s', pretrained=True)
        
        print('features:', self.efficientnetv2.classifier.in_features)
        self.efficientnetv2.classifier = nn.Linear(self.efficientnetv2.classifier.in_features, num_classes)

    def forward(self, x):
        return self.efficientnetv2(x)

class Trainer:
    def __init__(self, train_loader, test_loader, model, lr=1e-4, weight_decay=1e-9, weights=[3876/(1342+3876), 1342/(1342+3876)]):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights = torch.FloatTensor(weights).to(self.device) if weights else None
        # Initialize the model
        self.model = model
        # self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        # num_ftrs = self.model.fc.in_features  # Get the number of features in input to the final fully connected layer
        # self.model.fc = nn.Linear(num_ftrs, 2)  # Adjust the layer to have 2 output features
        self.model.to(self.device)
        self.lr = lr
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss(weight=self.weights)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)#, weight_decay=weight_decay)
        # self.optimizer = optim.RMSprop(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        # self.optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        # # Metrics
        self.accuracy_metric = Accuracy(task='multiclass', num_classes=2,average='macro').to(self.device)
        self.precision_metric = Precision(task='multiclass', num_classes=2, average='macro').to(self.device)
        self.recall_metric = Recall(task='multiclass', num_classes=2, average='macro').to(self.device)
        self.f1_metric = F1Score(task='multiclass', num_classes=2, average='macro').to(self.device)
        # self.accuracy_metric = Accuracy(threshold=0.5, task='binary').to(self.device)
        # self.precision_metric = Precision(threshold=0.5, task='binary').to(self.device)
        # self.recall_metric = Recall(threshold=0.5, task='binary').to(self.device)
        # self.f1_metric = F1Score(threshold=0.5, task='binary').to(self.device)

    def train(self, num_epochs=10):
        accs = []
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience=2)

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            
            # Reset metrics at the start of each epoch
            self.accuracy_metric.reset()
            self.precision_metric.reset()
            self.f1_metric.reset()
            self.recall_metric.reset()
            
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
                    self.recall_metric.update(outputs, labels)
                    
                    running_loss += loss.item()
                    tepoch.set_postfix(loss=running_loss/len(self.train_loader))
            
            # Compute metrics at the end of each epoch
            accuracy = self.accuracy_metric.compute()
            precision = self.precision_metric.compute()
            f1 = self.f1_metric.compute()
            recall = self.recall_metric.compute()

            accs.append(accuracy.item())
            
            # Print metrics at the end of each epoch
            print(f'Epoch {epoch+1}/{num_epochs} - Loss: {running_loss/len(self.train_loader)}, '
                f'Accuracy: {accuracy}, Precision: {precision}, F1 Score: {f1}, Recall: {recall}')

            res = self.test()
            self.scheduler.step(res['test_accuracy'])

        return {
            'train_accuracy': accuracy.item(),
            'train_precision': precision.item(),
            'train_recall': recall.item(),
            'train_f1': f1.item(),
            'train_acc_avg':np.array(accs).mean(),
            'train_acc_std':np.array(accs).std(),
            'train_sharpe':np.array(accs).mean()/np.array(accs).std(),
        }

    # def validate(self):
    #     self.model.eval()
    #     self.accuracy_metric.reset()
    #     self.precision_metric.reset()
    #     self.f1_metric.reset()
    #     self.recall_metric.reset()
        
    #     with torch.no_grad(), tqdm(self.val_loader, unit="batch") as tepoch:
    #         for images, labels in tepoch:
    #             tepoch.set_description("Validating")
                
    #             images, labels = images.to(self.device), labels.to(self.device)
    #             outputs = self.model(images)
                
    #             # Update metrics
    #             self.accuracy_metric.update(outputs, labels)
    #             self.precision_metric.update(outputs, labels)
    #             self.f1_metric.update(outputs, labels)
    #             self.recall_metric.update(outputs, labels)
            
    #         # Compute metrics
    #         accuracy = self.accuracy_metric.compute()
    #         precision = self.precision_metric.compute()
    #         f1 = self.f1_metric.compute()
    #         recall = self.recall_metric.compute()
            
    #         print(f'Validation - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, F1 Score: {f1:.2f}, Recall: {recall:.2f}')
    
    def test(self):
        self.model.eval()

        self.accuracy_metric.reset()
        self.precision_metric.reset()
        self.f1_metric.reset()
        self.recall_metric.reset()

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                
                # Update metrics
                self.accuracy_metric.update(outputs, labels)
                self.precision_metric.update(outputs, labels)
                self.f1_metric.update(outputs, labels)
                self.recall_metric.update(outputs, labels)

            # Compute metrics
            accuracy = self.accuracy_metric.compute()
            precision = self.precision_metric.compute()
            f1 = self.f1_metric.compute()
            recall = self.recall_metric.compute()


            print(f'Test - Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, F1 Score: {f1:.2f}, Recall: {recall:.2f}')

        return {
            'test_accuracy': accuracy.item(),
            'test_precision': precision.item(),
            'test_recall': recall.item(),
            'test_f1': f1.item()
        }
