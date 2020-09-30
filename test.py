import model
import torch
import torch.nn as nn
import torchvision
import os
import argparse

dev = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()

# Path
parser.add_argument('--dataset_name', type=str, default='IMAGENET-128')
parser.add_argument('--exp_version', type=str, default='FC_bias')

# Hyperparam
parser.add_argument('--z_dim', type=int, default=128)
parser.add_argument('--fc_bias', action='store_true')
parser.add_argument('--one_hot', action='store_true')

# Test setting
parser.add_argument('--load_iter', type=int, default=500000)
parser.add_argument('--save_img_num_per_class', type=int, default=10)
parser.add_argument('--divide', type=int, default=10)


config = parser.parse_args()

# Set output path
dir_result = os.path.join('output', config.dataset_name, config.exp_version, 'gen_imgs')
if not os.path.exists(dir_result):
    os.makedirs(dir_result)
path_result = os.path.join(dir_result, str(config.load_iter) + '.png')

if config.dataset_name == 'IMAGENET-128':
    num_classes = 1000

# Build Generator
print('Build model ...')
G = nn.DataParallel(model.Generator_128_v2(z_dim=config.z_dim,
                                           num_classes=num_classes,
                                           fc_bias=config.fc_bias)).to(dev)
file_name = 'ckpt_' + str(config.load_iter) + '.pkl'
path_ckpt = os.path.join('output', config.dataset_name, config.exp_version, 'weight', file_name)
ckpt = torch.load(path_ckpt)
G.load_state_dict(ckpt['G'])
G.eval()

with torch.no_grad():
    print('Image generating ...')
    result_imgs = []
    for cls_idx in range(num_classes):
        if cls_idx % 50 == 0:
            print('[%d / %d] ...' % (cls_idx + 1, num_classes))
        label = torch.LongTensor([cls_idx] * config.save_img_num_per_class)
        if config.one_hot:
            idx = label.unsqueeze(dim=1).to(dev)
            label = torch.zeros((label.size(0), num_classes)).to(dev)
            label.scatter_(1, idx, 1)
        z = torch.randn(config.save_img_num_per_class, config.z_dim).to(dev)
        result_imgs.append(G(z, label).cpu())
        
    result_imgs = torch.cat(result_imgs, dim=0)
    
    print('Image saving ...')
    for i in range(config.divide):
        start = i * int(num_classes / config.divide) * config.save_img_num_per_class
        end = (i + 1) * int(num_classes / config.divide) * config.save_img_num_per_class
        path_result = os.path.join(dir_result, str(config.load_iter) + '_' + str(i) + '.png')
        torchvision.utils.save_image(result_imgs[start:end], 
                                     path_result, 
                                     nrow=config.save_img_num_per_class,
                                     normalize=True)
