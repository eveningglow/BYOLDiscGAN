# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
import torchvision.models as models

from . import layer

class ResNet18(torch.nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.encoder = torch.nn.Sequential(*list(resnet.children())[:-1])
        
    def forward(self, x):
        h = self.encoder(x)
        return h.view(h.size(0), h.size(1))
    
    def extract_feature(self, x):
        h = self.encoder(x)
        return h.view(h.size(0), h.size(1))
    
class MoCo_Resnet_18(nn.Module):
    '''
    Resnet 18.

    Args:
        dim (int): Dimension of the last layer.
    '''
    def __init__(self, dim=128):
        super(MoCo_Resnet_18, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.fc = nn.Linear(512, dim)
        
    def forward(self, x):
        out = self.resnet(x)
        norm = torch.norm(out, p='fro', dim=1, keepdim=True)
        return out / norm
    
class MoCo_Resnet_50(nn.Module):
    '''
    Resnet 50.

    Args:
        dim (int): Dimension of the last layer.
    '''
    def __init__(self, dim=128):
        super(MoCo_Resnet_50, self).__init__()
        self.resnet = models.resnet50(pretrained=False)
        self.resnet.fc = nn.Linear(2048, dim)
        
    def forward(self, x):
        out = self.resnet(x)
        norm = torch.norm(out, p='fro', dim=1, keepdim=True)
        return out / norm

class Discriminator_32(nn.Module):
    def __init__(self):
        super(Discriminator_32, self).__init__()
        
        self.res_blocks = nn.Sequential(
            layer.D_InitResBlock(in_dim=3, out_dim=64),
            layer.D_ResBlock(in_dim=64, out_dim=128, downsampling=True),
            layer.D_ResBlock(in_dim=128, out_dim=256, downsampling=True),
            layer.D_ResBlock(in_dim=256, out_dim=512, downsampling=False)
        )
        self.linear = spectral_norm(nn.Linear(512, 1))

    def forward(self, x):
        out = self.extract_feature(x)
        out = self.linear(out)
        return out
    
    def extract_feature(self, x):
        out = self.res_blocks(x)
        out = F.relu(out)
        out = torch.sum(out, dim=(2, 3), keepdim=False)
        return out
    
class Generator_32(nn.Module):
    def __init__(self, z_dim):
        super(Generator_32, self).__init__()
        self.linear = nn.Linear(z_dim, 512 * 4 * 4)
        self.res_blocks = nn.Sequential(
            #G_ResBlock(in_dim=512, out_dim=512, num_classes=num_classes),
            layer.G_ResBlock(in_dim=512, out_dim=256),
            layer.G_ResBlock(in_dim=256, out_dim=128),
            layer.G_ResBlock(in_dim=128, out_dim=64)
        )
        self.out_blocks = nn.Sequential(
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        N = z.size(0)
        out = self.linear(z).view(N, 512, 4, 4)
        out = self.res_blocks(out)
        out = self.out_blocks(out)
        return out
    
class Discriminator_64(nn.Module):
    def __init__(self):
        super(Discriminator_64, self).__init__()
        
        self.res_blocks = nn.Sequential(
            layer.D_InitResBlock(in_dim=3, out_dim=64),
            layer.D_ResBlock(in_dim=64, out_dim=128, downsampling=True),
            layer.D_ResBlock(in_dim=128, out_dim=256, downsampling=True),
            layer.D_ResBlock(in_dim=256, out_dim=512, downsampling=True),
            layer.D_ResBlock(in_dim=512, out_dim=1024, downsampling=False)
        )
        
        self.linear = spectral_norm(nn.Linear(1024, 1))

    def forward(self, x):
        out = F.relu(self.res_blocks(x))
        out = torch.sum(out, dim=(2, 3), keepdim=False)
        out = self.linear(out)
        return out

class Generator_64(nn.Module):
    def __init__(self, z_dim):
        super(Generator_64, self).__init__()
        self.linear = nn.Linear(z_dim, 1024 * 4 * 4)
        self.res_blocks = nn.Sequential(
            layer.G_ResBlock(in_dim=1024, out_dim=512),
            layer.G_ResBlock(in_dim=512, out_dim=256),
            layer.G_ResBlock(in_dim=256, out_dim=128),
            layer.G_ResBlock(in_dim=128, out_dim=64)
        )
        self.out_blocks = nn.Sequential(
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        N = z.size(0)
        out = self.linear(z).view(N, 1024, 4, 4)
        out = self.res_blocks(out)
        out = self.out_blocks(out)
        return out
    
class Discriminator_128(nn.Module):
    def __init__(self):
        super(Discriminator_128, self).__init__()
        
        self.res_blocks = nn.Sequential(
            layer.D_InitResBlock(in_dim=3, out_dim=64),
            layer.D_ResBlock(in_dim=64, out_dim=128, downsampling=True),
            layer.D_ResBlock(in_dim=128, out_dim=256, downsampling=True),
            layer.D_ResBlock(in_dim=256, out_dim=512, downsampling=True),
            layer.D_ResBlock(in_dim=512, out_dim=1024, downsampling=True),
            layer.D_ResBlock(in_dim=1024, out_dim=1024, downsampling=False)
        )
        
        self.linear = spectral_norm(nn.Linear(1024, 1))

    def forward(self, x):
        out = F.relu(self.res_blocks(x))
        out = torch.sum(out, dim=(2, 3), keepdim=False)
        out = self.linear(out)
        return out
    
    def extract_feature(self, x):
        out = F.relu(self.res_blocks(x))
        out = torch.sum(out, dim=(2, 3), keepdim=False)
        return out
    
class AuxDiscriminator_128(nn.Module):
    def __init__(self, hidden_dim=512, projection_dim=128):
        super(AuxDiscriminator_128, self).__init__()
        
        self.res_blocks = nn.Sequential(
            layer.D_InitResBlock(in_dim=3, out_dim=64),
            layer.D_ResBlock(in_dim=64, out_dim=128, downsampling=True),
            layer.D_ResBlock(in_dim=128, out_dim=256, downsampling=True),
            layer.D_ResBlock(in_dim=256, out_dim=512, downsampling=True),
            layer.D_ResBlock(in_dim=512, out_dim=1024, downsampling=True),
            layer.D_ResBlock(in_dim=1024, out_dim=1024, downsampling=False)
        )
        
        self.adv_linear = spectral_norm(nn.Linear(1024, 1))
        
        self.byol_mlp = nn.Sequential(
                            nn.Linear(1024, hidden_dim),
                            nn.BatchNorm1d(hidden_dim),
                            nn.ReLU(inplace=True),
                            nn.Linear(hidden_dim, projection_dim)
        )
        
        self.byol_predictor = nn.Sequential(
                                nn.Linear(projection_dim, hidden_dim),
                                nn.BatchNorm1d(hidden_dim),
                                nn.ReLU(inplace=True),
                                nn.Linear(hidden_dim, projection_dim)
        )
        
    def forward(self, x):
        out = F.relu(self.res_blocks(x))
        out = torch.sum(out, dim=(2, 3), keepdim=False)
#         out = self.extract_feature(x)
        out = self.adv_linear(out)
        return out

    def extract_feature(self, x):
        out = nn.Sequential(* list(self.res_blocks.children())[:-2])(x)
        return out
    
    def extract_proj_feature(self, x):
        out = F.relu(self.res_blocks(x))
        out = torch.sum(out, dim=(2, 3), keepdim=False)
#         out = self.extract_feature(x)
        out = self.byol_mlp(out)
        return out
    
    def extract_pred_feature(self, x):
        out = F.relu(self.res_blocks(x))
        out = torch.sum(out, dim=(2, 3), keepdim=False)
        out = self.byol_mlp(out)
#         out = self.extract_proj_feature(x)
        out = self.byol_predictor(out)
        return out
    
    
class BYOLReconDiscriminator_128(nn.Module):
    def __init__(self, hidden_dim=512, projection_dim=128, recon_dim=128, feature_loc=-1):
        super(BYOLReconDiscriminator_128, self).__init__()
        self.feature_loc = feature_loc
        
        # Shared encoder with ResNet architecture
        self.res_blocks = nn.Sequential(
            layer.D_InitResBlock(in_dim=3, out_dim=64),
            layer.D_ResBlock(in_dim=64, out_dim=128, downsampling=True),
            layer.D_ResBlock(in_dim=128, out_dim=256, downsampling=True),
            layer.D_ResBlock(in_dim=256, out_dim=512, downsampling=True),
            layer.D_ResBlock(in_dim=512, out_dim=1024, downsampling=True),
            layer.D_ResBlock(in_dim=1024, out_dim=1024, downsampling=False)
        )
        
        # Linear layer for adversarial learning
        self.adv_linear = spectral_norm(nn.Linear(1024, 1))
        
        # MLP for BYOL framework
        self.byol_mlp = nn.Sequential(
                            nn.Linear(1024, hidden_dim),
                            nn.BatchNorm1d(hidden_dim),
                            nn.ReLU(inplace=True),
                            nn.Linear(hidden_dim, projection_dim)
        )
        
        self.byol_predictor = nn.Sequential(
                                nn.Linear(projection_dim, hidden_dim),
                                nn.BatchNorm1d(hidden_dim),
                                nn.ReLU(inplace=True),
                                nn.Linear(hidden_dim, projection_dim)
        )
        
        # Linear layer for image reconstruction
        self.ae_linear = nn.Linear(1024, recon_dim)
        
        # Auxiliary decoder for image reconstruction
        self.decoder = Generator_128(recon_dim)
        
    def forward(self, x, recon):
        shared = F.relu(self.res_blocks(x))
        shared = torch.sum(shared, dim=(2, 3), keepdim=False)
        adv_out = self.adv_linear(shared)
        
        if recon:
            img_recon = self.ae_linear(shared)
            img_recon = self.decoder(img_recon)
            return adv_out, img_recon

        else:
            return adv_out
        
    def extract_feature(self, x):
        if self.feature_loc == 0:
            out = F.relu(self.enc_res_blocks(x))
            out = torch.sum(out, dim=(2, 3), keepdim=False)
        else:
            out = nn.Sequential(* list(self.res_blocks.children())[:self.feature_loc])(x)
        return out
    
    def extract_pred_feature(self, x):
        out = F.relu(self.res_blocks(x))
        out = torch.sum(out, dim=(2, 3), keepdim=False)
        out = self.byol_mlp(out)
        out = self.byol_predictor(out)
        return out

class TargetNetwork_128(nn.Module):
    def __init__(self, hidden_dim=512, projection_dim=128):
        super(TargetNetwork_128, self).__init__()
        
        self.res_blocks = nn.Sequential(
            layer.D_InitResBlock(in_dim=3, out_dim=64),
            layer.D_ResBlock(in_dim=64, out_dim=128, downsampling=True),
            layer.D_ResBlock(in_dim=128, out_dim=256, downsampling=True),
            layer.D_ResBlock(in_dim=256, out_dim=512, downsampling=True),
            layer.D_ResBlock(in_dim=512, out_dim=1024, downsampling=True),
            layer.D_ResBlock(in_dim=1024, out_dim=1024, downsampling=False)
        )
        
        self.byol_mlp = nn.Sequential(
                            nn.Linear(1024, hidden_dim),
                            nn.BatchNorm1d(hidden_dim),
                            nn.ReLU(inplace=True),
                            nn.Linear(hidden_dim, projection_dim)
        )
        
    def forward(self, x_aug):        
        byol_feature = F.relu(self.res_blocks(x_aug))
        byol_feature = torch.sum(byol_feature, dim=(2, 3), keepdim=False)
        byol_feature = self.byol_mlp(byol_feature)
        return byol_feature
    
class Generator_128(nn.Module):
    def __init__(self, z_dim):
        super(Generator_128, self).__init__()
        self.linear = nn.Linear(z_dim, 1024 * 4 * 4)
        self.res_blocks = nn.Sequential(
            layer.G_ResBlock(in_dim=1024, out_dim=1024),
            layer.G_ResBlock(in_dim=1024, out_dim=512),
            layer.G_ResBlock(in_dim=512, out_dim=256),
            layer.G_ResBlock(in_dim=256, out_dim=128),
            layer.G_ResBlock(in_dim=128, out_dim=64)
        )
        self.out_blocks = nn.Sequential(
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        N = z.size(0)
        out = self.linear(z).view(N, 1024, 4, 4)
        out = self.res_blocks(out)
        out = self.out_blocks(out)
        return out