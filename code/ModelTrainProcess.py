from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import models
import time
import copy

class ModelTrain():
    def __init__(self, m_datasets):
        self.datasets = m_datasets
        # self.dataloaders = torch.utils.data.DataLoader(self.datasets, batch_size=4, shuffle=True)
        self.dataloaders = torch.utils.data.DataLoader(self.datasets, batch_size=4, shuffle=True, num_workers = 4)
        self.dataset_sizes = len(self.datasets)
        self.class_names = self.datasets.classes
        # self.device = torch.device("cpu")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def singleTrain_model(self, model, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            model.train()  # Set model to training mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in self.dataloaders:
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
                loss.backward()
                optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            scheduler.step()

            epoch_loss = running_loss / self.dataset_sizes
            epoch_acc = running_corrects.double() / self.dataset_sizes

            print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model

    def fitModel(self):
        inputs, classes = next(iter(self.dataloaders))
        out = torchvision.utils.make_grid(inputs)
        model_ft = models.resnet18(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        model_ft.fc = nn.Linear(num_ftrs, len(self.class_names))

        model_ft = model_ft.to(self.device)

        criterion = nn.CrossEntropyLoss()
        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

        model_ft = self.singleTrain_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                                     num_epochs=25)