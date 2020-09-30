# dataloader.py

import os
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as Transforms
import torchvision.datasets as Datasets
from torch.utils.data import Dataset, DataLoader

class GaussianBlur(object):
    """blur a single image on CPU"""

    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = Transforms.ToTensor()
        self.tensor_to_pil = Transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img

class BYOLTransform(object):
    def __init__(self, resize):
        self.resize = resize
        
    def __call__(self, pic):
        # Hard transform
        transform = Transforms.Compose([
                         Transforms.RandomResizedCrop(size=(self.resize, self.resize)),
                         Transforms.RandomHorizontalFlip(),
                         Transforms.RandomApply([Transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                         Transforms.RandomGrayscale(p=0.2),
                         #GaussianBlur(kernel_size=int(self.resize * 0.1)),
                         Transforms.ToTensor(),
                         Transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                              std=(0.5, 0.5, 0.5))
        ])
        
        # Weak transform
#         transform = Transforms.Compose([
#                          Transforms.RandomResizedCrop(size=(self.resize, self.resize)),
#                          Transforms.RandomHorizontalFlip(),
#                          Transforms.RandomApply([Transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)], p=0.8),
#                          Transforms.ToTensor(),
#                          Transforms.Normalize(mean=(0.5, 0.5, 0.5),
#                                               std=(0.5, 0.5, 0.5))
#         ])
        return transform(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'
    
class DefaultDataset(Dataset):
    def __init__(self, root, adv_transform=None, byol_transform=None):
        self.samples = os.listdir(root)
        self.samples.sort()
        self.samples = [os.path.join(root, img_name) for img_name in self.samples]
        
        self.adv_transform = adv_transform
        self.byol_transform = byol_transform
        
    def __getitem__(self, index):
        fname = self.samples[index]
        img = Image.open(fname).convert('RGB')
        
        img_aug_1 = self.byol_transform(img)
        img_aug_2 = self.byol_transform(img)
        img = self.adv_transform(img)
        return img, img_aug_1, img_aug_2

    def __len__(self):
        return len(self.samples)

def data_loader(dataset_root='/data_nfs/camist003/user/yunjey_nfs/', dataset_name='celeba_hq_images', resize=128, batch_size=64, num_worker=8, shuffle=True):
    
    adv_transform = Transforms.Compose([Transforms.Resize(size=resize),
                                        Transforms.ToTensor(),
                                        Transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                             std=(0.5, 0.5, 0.5))])
    
    byol_transform = BYOLTransform(resize=resize)
        
    # Prepare dataset
    if dataset_name == 'cifar-10-npz':
        dataset_root = os.path.join(dataset_root, dataset_name)
        dset = Datasets.CIFAR10(root=dataset_root, train=True, download=True, transform=transform)
    elif dataset_name == 'cifar-10':
        dataset_root = os.path.join(dataset_root, dataset_name, 'images')
        dset = DefaultDataset(root=dataset_root, transform=transform)
    elif dataset_name == 'celeba_hq_images':
        dataset_root = os.path.join(dataset_root, dataset_name, 'images')
        dset = DefaultDataset(root=dataset_root, adv_transform=adv_transform, byol_transform=byol_transform)
        
    dlen = len(dset)
    dloader = DataLoader(dataset=dset, batch_size=batch_size, shuffle=shuffle, num_workers=num_worker)
    return dloader, dlen