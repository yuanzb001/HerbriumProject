from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
import time
import copy

from ResNet50Model import ResNet50
from ResNet18Model import ResNet18

class ModelTrain():
    def __init__(self, m_datasets, ratio):
        self.datasets = m_datasets
        self.dataset_sizes = {}
        self.dataloaders = {}
        # self.dataloaders = torch.utils.data.DataLoader(self.datasets, batch_size=4, shuffle=True)
        self.dataset_sizes['train'] = int(len(self.datasets)*ratio)
        self.dataset_sizes['test'] = int(len(self.datasets)) - int(len(self.datasets)*ratio)
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(self.datasets,[self.dataset_sizes['train'], self.dataset_sizes['test']], generator=torch.Generator().manual_seed(0))
        self.dataloaders['train'] = torch.utils.data.DataLoader(self.train_dataset, batch_size=128, shuffle=True, num_workers=16)
        self.dataloaders['test'] = torch.utils.data.DataLoader(self.test_dataset, batch_size=32, shuffle=True, num_workers=16)
        self.class_names = self.datasets.classes
        # self.device = torch.device("cpu")
        self.device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    def train_model(self, model, criterion, optimizer, num_epochs=30):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)
            for phase in ['train', 'test']:
                # Each epoch has a training and test phase
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                index = 0
                # Iterate over data.
                for inputs, labels in self.dataloaders[phase]:
                    index += 1
                    inputs = inputs.to(self.device)
                    labels = torch.tensor(labels).to(self.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                   
                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                # deep copy the model
                if phase == 'test' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model

    def fitModel(self, type):
        if type == 'ResNet50':
            model_ft = ResNet50(len(self.class_names))
        elif type == 'ResNet18':
            model_ft = ResNet18(len(self.class_names))
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, len(self.class_names))
        model_ft = model_ft.to(self.device)

        criterion = nn.CrossEntropyLoss()
        # Observe that all parameters are being optimized
        optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)
        model_ft = self.train_model(model_ft, criterion, optimizer_ft,num_epochs=300)
        return model_ft
