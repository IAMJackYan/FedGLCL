import time
import os
import logging
import torch
import numpy as np

from pathlib import Path
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


def set_for_logger(args):

    log_filename = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()) + '.txt'
    log_filepath = os.path.join(args.log_dir, log_filename)

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s ===> %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # args FileHandler to save log file
    fh = logging.FileHandler(log_filepath)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    # args StreamHandler to print log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    # add the two Handler
    logger.addHandler(ch)
    logger.addHandler(fh)



class OfficeDataset(Dataset):
    def __init__(self, base_path, site, train=True, transform=None):
        if train:
            self.paths, self.text_labels = np.load('./data/office_caltech_10/{}_train.pkl'.format(site), allow_pickle=True)
        else:
            self.paths, self.text_labels = np.load('./data/office_caltech_10/{}_test.pkl'.format(site), allow_pickle=True)
            
        label_dict={'back_pack':0, 'bike':1, 'calculator':2, 'headphones':3, 'keyboard':4, 'laptop_computer':5, 'monitor':6, 'mouse':7, 'mug':8, 'projector':9}
        self.labels = [label_dict[text] for text in self.text_labels]
        self.transform = transform
        self.base_path = base_path if base_path is not None else '../data'

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.paths[idx])
        label = self.labels[idx]
        image = Image.open(img_path)

        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class DomainNetDataset(Dataset):
    def __init__(self, base_path, site, train=True, transform=None):
        if train:
            self.paths, self.text_labels = np.load('./data/DomainNet/{}_train.pkl'.format(site), allow_pickle=True)
        else:
            self.paths, self.text_labels = np.load('./data/DomainNet/{}_test.pkl'.format(site), allow_pickle=True)
            
        label_dict = {'bird':0, 'feather':1, 'headphones':2, 'ice_cream':3, 'teapot':4, 'tiger':5, 'whale':6, 'windmill':7, 'wine_glass':8, 'zebra':9}     
        
        self.labels = [label_dict[text] for text in self.text_labels]
        self.transform = transform
        self.base_path = base_path if base_path is not None else '../data'

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.paths[idx])
        label = self.labels[idx]
        image = Image.open(img_path)
        
        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)
            print(img_path)

        if self.transform is not None:
            image = self.transform(image)

        return image, label
