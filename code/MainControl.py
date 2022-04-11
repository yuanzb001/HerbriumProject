from __future__ import print_function, division
from torchvision import transforms
import os
import torch

from MyDataLoader import myDataSet
from DataExtraction import DataExtraction
from ModelTrainProcess import ModelTrain

dataUtil = DataExtraction('/pless_nfs/home/yuanzhuobin/project/Herbarium/data/train_metadata.json')
fileNameList, label_list = dataUtil.getImagePathandLabel()

#print(fileNameList)

data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

data_dir = '/pless_nfs/home/yuanzhuobin/project/Herbarium/data/train_images'

image_datasets = myDataSet(data_dir, fileNameList, label_list, data_transforms)
#print(image_datasets)

m_modelTrain = ModelTrain(image_datasets, ratio = 0.7)
best_model = m_modelTrain.fitModel('ResNet18')
