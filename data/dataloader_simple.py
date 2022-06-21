import os

import torch
import torchvision.transforms as T
from torchvision.datasets import CIFAR100
from numpy import loadtxt
import pandas as pd
from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from PIL import Image
from typing import Tuple
import random

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

test_cifar_10_transform= T.Compose([
            T.ToPILImage(),
            T.ToTensor(),
            T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ])
test_cifar_100_transform= T.Compose([
            T.ToPILImage(),
            T.ToTensor(),
            T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ])

train_cifar_10_transform = T.Compose([#T.ToPILImage(),
                                      #T.RandomCrop(32, padding=4),
                                      #T.RandomHorizontalFlip(),
                                      #T.RandomVerticalFlip(p=0.5),
                                      T.ToTensor(),
                                      T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
                                      ])

train_cifar_100_transform = T.Compose([
            #T.ToPILImage(),
            #T.RandomCrop(32, padding=4),
            #T.RandomHorizontalFlip(),
            #T.RandomVerticalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ])

class MyCifar10(CIFAR10):
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        super(MyCifar10, self).__init__(root, train, transform,
                                        target_transform, download)
        self.num_coarse_labels = 2
        self.num_sub_labels = 10

    def __getitem__(self, index: int) -> Tuple[type(Image), int, type(Image)]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])
        if self.transform is not None:
            img = self.transform(img)
        coarse10_labels = [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
        superclass = coarse10_labels[target]

        return img, target, superclass


class Cifar10pseudo(CIFAR10):
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        super(Cifar10pseudo, self).__init__(root, train, transform,
                                        target_transform, download)

        self.num_coarse_labels = 2
        self.subclass = self.targets
        self.add_labels(train)
        self.num_sub_labels = pd.Series(self.subclass).nunique()-1

    def __getitem__(self, index: int) -> Tuple[type(Image), int, type(Image)]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, sub, target = self.data[index],int(self.subclass[index]), int(self.targets[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, sub, target

    def add_labels(self, train):
        """
        Adds pseudo-labels
        """
        pseudo = loadtxt('sub_cifar10' + '\pseudo.csv', delimiter=',')
        if(train==True):
            superclass = loadtxt('sub_cifar10'+'\superclassi.csv', delimiter=',')
            self.subclass = pseudo
            self.targets = superclass
        else:
            coarse10_labels = [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
            self.subclass = [x*0 for x in self.targets]
            self.targets = [coarse10_labels[x] for x in self.targets]

class MyCifar100(CIFAR100):
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        super(MyCifar100, self).__init__(root, train, transform,
                                         target_transform, download)
        self.num_coarse_labels = 20
        self.num_sub_labels = 100
        self.coarse_labels = torch.Tensor([4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                                           3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                                           6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                                           0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                                           5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                                           16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                                           10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                                           2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                                           16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                                           18, 1, 2, 15, 6, 0, 17, 8, 14, 13])

    def __getitem__(self, index: int) -> Tuple[type(Image), int]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, target,int(self.coarse_labels[target])

class Cifar100pseudo(CIFAR100):
    def __init__(self, root, train=True, transform=None,
                 target_transform=None, download=False) -> None:
        super(Cifar100pseudo, self).__init__(root, train, transform,
                                        target_transform, download)

        self.num_coarse_labels = 20
        self.coarse_labels = torch.Tensor([4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                                           3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                                           6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                                           0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                                           5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                                           16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                                           10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                                           2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                                           16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                                           18, 1, 2, 15, 6, 0, 17, 8, 14, 13])
        self.subclass = self.targets
        self.add_labels(train)
        self.num_sub_labels = pd.Series(self.subclass).nunique()-1


    def __getitem__(self, index: int) -> Tuple[type(Image), int, int]:
        """
        Gets the requested element from the dataset.
        :param index: index of the element to be returned
        :returns: tuple: (image, target) where target is index of the target class.
        """
        img, sub, target = self.data[index],int(self.subclass[index]), int(self.targets[index])

        if self.transform is not None:
            img = self.transform(img)

        return img, sub, target

    def add_labels(self, train):
        """
        Adds pseudo-labels
        """
        pseudo = loadtxt('sub_cifar100' + '\pseudo.csv', delimiter=',')
        if train==True:
            superclass = loadtxt('sub_cifar100'+'\superclassi.csv', delimiter=',')
            self.subclass = pseudo
            self.targets = superclass
        else:
            self.subclass = [x*0 for x in self.targets]
            self.targets = [self.coarse_labels[x] for x in self.targets]


def load_cifar10(batch_size=128, pseudo_labels=False, not_shuffle=False):
    prop_train=0.9
    if not_shuffle:
        trainset = MyCifar10(root=ROOT_DIR + '/CIFAR10', train=True,
                             download=True, transform=train_cifar_10_transform)
        trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=0, shuffle=False)
        return trainset,trainloader
    if pseudo_labels:
        trainset = Cifar10pseudo(root=ROOT_DIR + '/CIFAR10', train=True,
                             download=True, transform=train_cifar_10_transform)
        valset = Cifar10pseudo(root=ROOT_DIR + '/CIFAR10', train=True,
                               download=True, transform=test_cifar_10_transform)
        testset = Cifar10pseudo(root=ROOT_DIR + '/CIFAR10', train=False,
                            download=True, transform=test_cifar_10_transform)

        mask = loadtxt('sub_cifar10'+'\mask.csv', delimiter=',')
        mask = torch.tensor(mask)
        idx = list()
        for i in torch.nonzero(mask == 0):
            idx.append(i)
        random.shuffle(idx)
        indices_train = idx[:int(prop_train * len(idx))]
        indices_val = idx[int(prop_train * len(idx)):]
        train_set = torch.utils.data.Subset(trainset, indices_train)
        val_set = torch.utils.data.Subset(trainset, indices_val)

        trainloader = DataLoader(train_set, batch_size=batch_size,num_workers=0)
        valloader = DataLoader(val_set, batch_size=batch_size,shuffle=False, num_workers=0)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    else:
        trainset = MyCifar10(root=ROOT_DIR + '/CIFAR10', train=True,
                             download=True,transform=train_cifar_10_transform)
        valset = MyCifar10(root=ROOT_DIR + '/CIFAR10', train=True,
                           download=True, transform=test_cifar_10_transform)
        testset = MyCifar10(root=ROOT_DIR + '/CIFAR10', train=False,
                            download=True, transform=test_cifar_10_transform)
        train_set, _ = torch.utils.data.random_split(trainset, [int(prop_train * len(trainset)),int(len(trainset)-int(prop_train * len(trainset)))])
        _, val_set = torch.utils.data.random_split(valset, [int(prop_train * len(valset)),int(len(valset)-int(prop_train * len(valset)))])
        trainloader = DataLoader(train_set, batch_size=batch_size,
                             shuffle=True, num_workers=0)
        valloader = DataLoader(val_set, batch_size=batch_size,
                               shuffle=False, num_workers=0)
        testloader = DataLoader(testset, batch_size=batch_size,
                                shuffle=False, num_workers=0)
    return trainset, trainloader, valset, valloader, testset, testloader

def load_cifar100(batch_size=128, not_shuffle=False, pseudo_labels=False):
    prop_train = 0.9
    if not_shuffle:
        trainset = MyCifar100(root=ROOT_DIR + '/CIFAR100', train=True,
                             download=True, transform=train_cifar_100_transform)
        trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=0, shuffle=False)
        return trainset, trainloader
    if pseudo_labels:
        trainset = Cifar100pseudo(root=ROOT_DIR + '/CIFAR100', train=True,
                             download=True, transform=train_cifar_100_transform)
        valset = Cifar100pseudo(root=ROOT_DIR + '/CIFAR100', train=True,
                               download=True, transform=test_cifar_100_transform)
        testset = Cifar100pseudo(root=ROOT_DIR + '/CIFAR100', train=False,
                            download=True, transform=test_cifar_100_transform)
        train_set, _ = torch.utils.data.random_split(trainset, [int(prop_train * len(trainset)),
                                                                int(len(trainset) - int(prop_train * len(trainset)))])
        _, val_set = torch.utils.data.random_split(valset, [int(prop_train * len(valset)),
                                                            int(len(valset) - int(prop_train * len(valset)))])
        mask = loadtxt('sub_cifar100' + '\mask.csv', delimiter=',')
        mask = torch.tensor(mask)
        idx = list()
        for i in torch.nonzero(mask == 0):
            idx.append(i)
        random.shuffle(idx)
        indices_train = idx[:int(prop_train * len(idx))]
        indices_val = idx[int(prop_train * len(idx)):]
        train_set = torch.utils.data.Subset(trainset, indices_train)
        val_set = torch.utils.data.Subset(trainset, indices_val)
        trainloader = DataLoader(train_set, batch_size=batch_size,
                             shuffle=True, num_workers=0)
        valloader = DataLoader(val_set, batch_size=batch_size,
                               shuffle=False, num_workers=0)
        testloader = DataLoader(testset, batch_size=batch_size,
                                shuffle=False, num_workers=0)
    else:
        trainset = MyCifar100(root=ROOT_DIR + '/CIFAR100', train=True,
                             download=True, transform=train_cifar_100_transform)
        valset = MyCifar100(root=ROOT_DIR + '/CIFAR100', train=True,
                               download=True, transform=test_cifar_100_transform)
        testset = MyCifar100(root=ROOT_DIR + '/CIFAR100', train=False,
                            download=True, transform=test_cifar_100_transform)

        train_set, _ = torch.utils.data.random_split(trainset, [int(prop_train * len(trainset)),
                                                                int(len(trainset) - int(prop_train * len(trainset)))])
        _, val_set = torch.utils.data.random_split(valset, [int(prop_train * len(valset)),
                                                            int(len(valset) - int(prop_train * len(valset)))])
        trainloader = DataLoader(train_set,batch_size=batch_size, shuffle=True, num_workers=0)
        valloader = DataLoader(val_set,batch_size=batch_size,shuffle=False, num_workers=0)
        testloader = DataLoader(testset, batch_size=batch_size,
                                shuffle=False, num_workers=0)
    return trainset, trainloader, valset, valloader, testset, testloader
