import torch
import torchvision.transforms as transforms
import torchvision
import os
from torchvision.transforms import transforms
from torchvision.datasets import CIFAR10
from torchvision.datasets import STL10
from PIL import Image
import cv2
import random
import numpy as np


def get_dataloader(opt):
    base_folder = os.path.join(opt.data_input_dir)
    if opt.dataset == 'cifar10':
        # transform_train = transforms.Compose([
        #     transforms.ToTensor()
        # ])
        transform_train = trans_cifar()

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = CIFAR10Pair(root=base_folder, train=True, transform=transform_train, download=True)
        supervised_dataset = torchvision.datasets.CIFAR10(root=base_folder, train=True, transform=transform_test, download=True)

        test_dataset = torchvision.datasets.CIFAR10(root=base_folder, train=False, download=False, transform=transform_test)
    elif opt.dataset == 'stl10':
        transform_train = trans_stl()

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4313, 0.4156, 0.3663), (0.2683, 0.2610, 0.2687)),
        ])

        train_dataset = STL10Pair(root=base_folder, split="unlabeled", transform=transform_train, download=False)
        supervised_dataset = torchvision.datasets.STL10(root=base_folder, split="train", transform=transform_test, download=False)

        test_dataset = torchvision.datasets.STL10(root=base_folder, split="test", download=False,
                                                    transform=transform_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                                  num_workers=16, pin_memory=True)

    supervised_loader = torch.utils.data.DataLoader(supervised_dataset, batch_size=opt.batch_size, shuffle=True,
                                  num_workers=16, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=16)

    return (
        train_loader,
        train_dataset,
        supervised_loader,
        supervised_dataset,
        test_loader,
        test_dataset,
    )


class CIFAR10Pair(CIFAR10):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target


class STL10Pair(STL10):
    def __getitem__(self, index):

        if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], None
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        img_1, img_2 = trans_crop_1(img)
        # img_1, img_2 = img, img

        if self.transform is not None:
            pos_1 = self.transform(img_1)
            pos_2 = self.transform(img_2)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target



def trans_cifar():
    trans = []
    trans.append(transforms.RandomResizedCrop(32))
    trans.append(transforms.RandomHorizontalFlip(p=0.5))
    # rand = transforms.RandomRotation(10)
    # trans.append(transforms.RandomApply([rand], p=0.5))
    color_dis = get_color_distortion(s=0.5)
    trans.append(color_dis)
    trans.append(transforms.ToTensor())
    trans.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
    trans = transforms.Compose(trans)
    return trans


def trans_crop_1(img):
    crop = transforms.RandomCrop(64)
    img_crop = crop(img)
    patch = transforms.RandomCrop(16)
    img_patch_1 = patch(img_crop)
    img_patch_2 = patch(img_crop)
    return img_patch_1, img_patch_2


def trans_crop_2(img):
    crop = transforms.RandomCrop(80)
    img_crop = crop(img)
    patch = transforms.RandomCrop(48)
    img_patch_1 = patch(img_crop)
    img_patch_2 = patch(img_crop)
    return img_patch_1, img_patch_2


def trans_stl():
    trans = []
    trans.append(transforms.RandomResizedCrop(16))
    trans.append(transforms.RandomHorizontalFlip(p=0.5))
    color_dis = get_color_distortion(s=0.5)
    trans.append(color_dis)
    # gaussian_blur = Gaussian_blur(kernel=9)
    # trans.append(gaussian_blur)
    trans.append(transforms.ToTensor())
    trans.append(transforms.Normalize((0.4313, 0.4156, 0.3663), (0.2683, 0.2610, 0.2687)))
    trans = transforms.Compose(trans)
    return trans


def get_color_distortion(s=1.0):
    # s is the strength of color distortion
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort


class Gaussian_blur(object):
    def __init__(self, kernel):
        self.kernel = kernel

    def __call__(self, img):
        a = random.randint(0, 1)
        if a == 0:
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            sigma = random.uniform(0.1, 2.0)
            img = cv2.GaussianBlur(img, (self.kernel, self.kernel), sigma)
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return img