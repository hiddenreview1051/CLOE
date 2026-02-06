#!/usr/bin/env python3
# https://github.com/HangtingYe/DRL/blob/main/utils.py
# -*- coding: utf-8 -*-
import csv
import logging
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import os
import sklearn
import torch

def aucPerformance(score, labels):
    roc_auc = roc_auc_score(labels, score)
    ap = average_precision_score(labels, score)
    return roc_auc, ap

def F1Performance(score, target):
    normal_ratio = (target == 0).sum() / len(target)
    score = np.squeeze(score)
    threshold = np.percentile(score, 100 * normal_ratio)
    pred = np.zeros(len(score))
    pred[score > threshold] = 1
    precision, recall, f1, _ = precision_recall_fscore_support(target, pred, average='binary')
    return f1


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "%(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

# https://github.com/HangtingYe/DRL/blob/main/DataSet/MyDataset.py
class NpzDataset(Dataset):
    def __init__(self, dataset_name: str, data_dim: int, data_dir: str, preprocess: str, mode: str = 'train', random_seed: int = 49):
        super(NpzDataset, self).__init__()
        path = os.path.join(data_dir, dataset_name+'.npz')
        data = np.load(path)  
        samples = data['X']
        labels = ((data['y']).astype(int)).reshape(-1)

        inliers = samples[labels == 0]
        outliers = samples[labels == 1]
        train_data, train_label, test_data, test_label = train_test_split_DRL(inliers, outliers, dataset_name, random_seed)
        
        if mode == 'train':
            self.data = torch.Tensor(train_data)
            self.targets =torch.Tensor(train_label)
        else:
            self.data = torch.Tensor(test_data)
            self.targets = torch.Tensor(test_label)

    def __getitem__(self, item):
        return self.data[item], self.targets[item]

    def __len__(self):
        return len(self.data)

def train_test_split_DRL(inliers, outliers, dataset_name, random_seed):
    if dataset_name == "9_census":
        test_size = 1- 3000/inliers.shape[0]
    elif dataset_name == "24_mnist":
        test_size = 1- 5000/inliers.shape[0]
    elif  inliers.shape[0]<8000:
        test_size = 0.1
    else:
        test_size = 1- 8000/inliers.shape[0]
    X_train_valid, X_test= train_test_split(inliers, test_size=test_size, random_state=random_seed)
    X_train, X_valid = train_test_split(X_train_valid,test_size=0.2, random_state=random_seed)
    train_label = np.zeros(X_train.shape[0])
    
    test_data = np.concatenate([inliers, outliers], 0)
    test_label = np.zeros(test_data.shape[0])
    test_label[-len(outliers):] = 1

    return X_train, train_label, test_data, test_label