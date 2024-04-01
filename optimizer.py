import optuna
import torch
import os
from dataloader import ChestXrayDatasetInMemory, DataLoader
from model.ResNet import resnet34
from model.VGG16 import VGG16
from model.AlexNet import AlexNet
from train import Trainer

dataset_path = 'data/chest_xray'
train_dataset = ChestXrayDatasetInMemory(os.path.join(dataset_path, 'train'), aug=True)
test_dataset = ChestXrayDatasetInMemory(os.path.join(dataset_path, 'test'), aug=False)

def objective(trial):
    model_type = trial.suggest_categorical('model_type', ['ResNet34','CustomPanda'])
    
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
    
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3)
    
    aug = trial.suggest_categorical('aug', [True, False])
    
    # Setup model
    if model_type == 'ResNet34':
        model = resnet34()
    elif model_type == 'VGG16':
        model = VGG16()
    elif model_type == 'AlexNet':
        model = AlexNet()

    
    # Load datasets with or without augmentation for the training set
    train_dataset.set_aug(aug)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    trainer = Trainer(train_loader, test_loader, model=model)
    
    train_metrics = trainer.train(num_epochs=3)  
    test_metrics = trainer.test() 
    
    # log all metrics with trial.set_user_attr for later analysis
    for key, value in {**train_metrics, **test_metrics}.items():
        trial.set_user_attr(key, value)

    return (train_metrics['train_sharpe'], test_metrics['test_accuracy'])  

storage_url = "postgresql://postgres: @localhost/optuna_chestxray"
study_name = "multi-rn34-panda"

study = optuna.create_study(
    study_name=study_name,
    storage=storage_url,
    directions=["maximize","maximize"],
    load_if_exists=True
)
study.optimize(objective, n_trials=100) 

print(f"Best trial: {study.best_trial.value}")
print(f"Best hyperparameters: {study.best_trial.params}")
