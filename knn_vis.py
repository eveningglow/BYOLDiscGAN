import sys
import os
import pickle
from collections import OrderedDict
from distutils.util import strtobool
import argparse
from PIL import Image

#from dataloader import data_loader
import model
from network import net
import util
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as Transforms
from torchvision.utils import save_image

import torchvision.datasets as Datasets
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
#import torchvision.transforms.functional as F

import pickle


import math
import random


class BYOLTransform(object):
    def __init__(self, resize, crop):
        self.resize = resize
        self.crop = crop
        
    def __call__(self, pic):
        random_resize = random.randint(self.crop, self.resize)
        transform = Transforms.Compose([
                         Transforms.Resize(size=random_resize),
                         Transforms.RandomCrop(size=self.crop),
                         Transforms.RandomHorizontalFlip(),
#                          Transforms.RandomApply([Transforms.ColorJitter(0.2, 0.2, 0.2, 0)], p=0.8),
                         Transforms.ToTensor(),
                         Transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                              std=(0.5, 0.5, 0.5))
        ])
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

def data_loader(dataset_root='/data_nfs/camist003/user/yunjey_nfs/', dataset_name='celeba_hq_images', resize=128, crop=128, batch_size=64, num_worker=8, shuffle=True):
    
    adv_transform = Transforms.Compose([Transforms.Resize(size=crop),
                                        Transforms.ToTensor(),
                                        Transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                             std=(0.5, 0.5, 0.5))])
    
    byol_transform = BYOLTransform(resize=resize, crop=crop)
        
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

# Set device and random seed
torch.manual_seed(666)

dev = 'cpu'
if torch.cuda.is_available():
    dev = 'cuda'
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
    
to_pil = Transforms.ToPILImage()

# Hyperparameter for debugging
weight_root = 'gen_result/KR62512_yunjey_nfs_459/app/expr/BYOLDiscGAN/celeba_hq_images/weight'
enc_weight_root = 'gen_result/KR62512_yunjey_nfs_459/app/expr/BYOLDiscGAN/celeba_hq_images/weight'
knn_root = 'gen_result/KR62512_yunjey_nfs_459/app/knn'
k = 10
z_dim = 128 # 이걸 내가 정할일이 있나?
save_iter = 100
queue_size = 2048
knn_type = 'precision'

if not os.path.exists(knn_root):
    os.makedirs(knn_root)

# Hyperparameter for setting
batch_size = 64
dataset_root = '/data_nfs/camist003/user/yunjey_nfs/'
dataset_name = 'celeba_hq_images'
multi_gpu = False

list_iter = [300000, 350000, 400000]
location = -1

for iters in list_iter:

    ''' ################################# Load model ################################# '''
    D = net.AuxDiscriminator_128(hidden_dim=2048, 
                                 projection_dim=128).to(dev)
    ckpt_path = os.path.join(weight_root, 'ckpt_' + str(iters) + '.pkl')
    ckpt = torch.load(ckpt_path)
    D.load_state_dict(ckpt['D'])

    if location < 0:
        D = nn.Sequential(* list(D.res_blocks.children())[:-location])




    # D = net.MoCo_Resnet_18(dim=128).to(dev)
    # if multi_gpu:
    #     net = torch.nn.DataParallel(net)

    # ckpt_path = os.path.join(enc_weight_root, 'mid_MOCO_ckpt.pkl')
    # ckpt = torch.load(ckpt_path)
    # D.load_state_dict(ckpt['encoder'])

    # D = nn.Sequential(* list(D.resnet.children())[:-4])






    # D = net.MoCo_Resnet_18(dim=128).to(dev)
    # if multi_gpu:
    #     net = torch.nn.DataParallel(net)

    # ckpt_path = os.path.join(enc_weight_root, 'mid_MOCO_ckpt.pkl')
    # ckpt = torch.load(ckpt_path)
    # D.load_state_dict(ckpt['encoder'])

    # encoder = D.resnet

    # 2x2_deep
    # sub_encoder = [encoder.conv1, encoder.bn1, encoder.relu, encoder.maxpool, 
    #                encoder.layer1, encoder.layer2, list(encoder.layer3.children())[0],
    #                list(encoder.layer3.children())[1].conv1]

    # 2x2
    # sub_encoder = [encoder.conv1, encoder.bn1, encoder.relu, encoder.maxpool, 
    #                encoder.layer1, encoder.layer2, list(encoder.layer3.children())[0].conv1]

    # 4x4 deep
    # sub_encoder = [encoder.conv1, encoder.bn1, encoder.relu, encoder.maxpool, 
    #               encoder.layer1, list(encoder.layer2.children())[0], list(encoder.layer2.children())[1].conv1]

    # 4x4
    # sub_encoder = [encoder.conv1, encoder.bn1, encoder.relu, encoder.maxpool, 
    #               encoder.layer1, list(encoder.layer2.children())[0].conv1]

    # D = nn.Sequential(*sub_encoder)



    ''' ################################# Build data loader ################################# '''
    print('Build dataloader ...')
    # query_dloader, query_data_len = data_loader(dataset_root=dataset_root, 
    #                                             dataset_name=dataset_name, 
    #                                             batch_size=1, 
    #                                             num_worker=1)

    dloader, dlen = data_loader(dataset_root=dataset_root, 
                                dataset_name=dataset_name, 
                                batch_size=64, 
                                num_worker=1)


    # precision : query - gen_img / queue - real_img
    # recall : query - real_img / queue - gen_img
    # real : query - real_img / queue - real_img

    real_img = []
    real_feature = []

    with torch.no_grad():
        for i, (img, _, _) in enumerate(dloader):
            
            if location < 0:
                feature = D(img.to(dev)).detach()
                feature = feature.view(feature.size(0), -1)
            else:
                feature = D.extract_feature(img.to(dev)).detach()

            real_img.append(img)
            real_feature.append(feature)

            if i == (queue_size / batch_size) - 1:
                break

        real_img = torch.cat(real_img, dim=0)
        real_feature = torch.cat(real_feature, dim=0)

        if knn_type == 'precision' or knn_type == 'recall':
            G = net.Generator_128(z_dim).to(dev)
            G.eval()

            ckpt_path = os.path.join(weight_root, 'ckpt_' + str(iters) + '.pkl')
            ckpt = torch.load(ckpt_path)
            G.load_state_dict(ckpt['G'])

            try:
                with open('z_knn.pkl', 'rb') as f:
                    z = pickle.load(f)[:128].to(dev)
            except:
                z = torch.randn(128, z_dim).to(dev)
                print('Cannot load z')

            fake_img = G(z)

    #         fake_feature = D.extract_feature(fake_img).detach()

            if location < 0:
                fake_feature = D(fake_img).detach()
                fake_feature = fake_feature.view(fake_feature.size(0), -1)
            else:
                fake_feature = D.extract_feature(fake_img).detach()
                
        if knn_type == 'precision':
            query_img = fake_img.cpu()
            query_feature = fake_feature

            queue_img = real_img
            queue_feature = real_feature

        elif knn_type == 'recall':
            query_img = real_img
            query_feature = real_feature

            queue_img = fake_img.cpu()
            queue_feature = fake_feature

        elif knn_type == 'real':
            query_img = real_img
            query_feature = real_feature

            queue_img = real_img
            queue_feature = real_feature

    #query = {'img': query_img, 'feature':query_feature}
    #queue = {'img': queue_img, 'feature': queue_feature}

    ''' ################################# KNN ################################# '''
    with torch.no_grad():
        result_imgs = []
        for i in range(query_feature.size(0)):
            query_feature_i = query_feature[i].unsqueeze(dim=0)
            query_img_i = query_img[i].unsqueeze(dim=0)

            dist = torch.sum((query_feature_i - queue_feature)**2, dim=1, keepdim=True)

            val, idx = torch.topk(dist, k, dim=0, largest=False, sorted=True)

            # Get corresponding min-k imgs
            top_k_img = queue_img[idx.squeeze(dim=1)]

            if knn_type == 'real':
                result_imgs.append(torch.cat([query_img_i.cpu(), top_k_img[1:]], dim=0))
            else:
                result_imgs.append(torch.cat([query_img_i.cpu(), top_k_img], dim=0))

            if (i > 0) and (i % save_iter == 0):
                result_imgs = torch.cat(result_imgs, dim=0) / 2 + 0.5
                file_name = 'GAN_iters_' + str(iters) + '_' + knn_type + '_' + str(int(i / save_iter)) + '.png'
#                 file_name = knn_type + '_' + str(int(i / save_iter)) + '.png'
                if knn_type == 'real':
                    save_image(result_imgs, os.path.join(knn_root, file_name), nrow=k, padding=0)
                else:
                    save_image(result_imgs, os.path.join(knn_root, file_name), nrow=k+1, padding=0)
                result_imgs = []
                print('Save ' + file_name)