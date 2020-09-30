# model.py
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm

class MultiInputSequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs
    
# 추후에 shared embedding을 사용하면 weight를 상위 단계의 Generator에서 
# Input이 one-hot이 아닌 index integer
class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_classes, num_features):
        super(ConditionalBatchNorm2d, self).__init__()
        
        self.bn = nn.BatchNorm2d(num_features, eps=2e-5, affine=False)
        
        self.gamma = nn.Embedding(num_classes, num_features)
        #self.gamma.weight.data.normal_(1, 0.02)
        self.gamma.weight.data.fill_(1)
        self.beta = nn.Embedding(num_classes, num_features)
        self.beta.weight.data.zero_()

    def forward(self, x, y):
        N, C = x.size(0), x.size(1)
        out = self.bn(x)
        gamma = self.gamma(y) #(N, num_features)
        beta = self.beta(y)   #(N, num_features)
        out = gamma.view(N, C, 1, 1) * out + beta.view(N, C, 1, 1)

        return out
    
class G_ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(G_ResBlock, self).__init__()

        self.upconv = nn.Sequential(
                            nn.BatchNorm2d(in_dim),
                            nn.ReLU(inplace=True),
                            nn.Upsample(scale_factor=2),
                            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)
        )

        self.conv = nn.Sequential(
                            nn.BatchNorm2d(out_dim),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        )

        self.short_cut = nn.Sequential(
                            nn.Upsample(scale_factor=2),
                            nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0)
        )      

    def forward(self, x):
        out = self.upconv(x)
        out = self.conv(out)
        
        return out + self.short_cut(x)
    
class G_ResBlock_cond(nn.Module):
    def __init__(self, in_dim, out_dim, num_classes):
        super(G_ResBlock_cond, self).__init__()
        
        # CBN - RELU - UP - CONV
        self.cbn_1 = ConditionalBatchNorm2d(num_classes, in_dim)
        self.upconv = nn.Sequential(
                            nn.ReLU(inplace=True),
                            nn.Upsample(scale_factor=2),
                            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)
        )

        # CBN - RELU - CONV
        self.cbn_2 = ConditionalBatchNorm2d(num_classes, out_dim)
        self.conv = nn.Sequential(
                            nn.ReLU(inplace=True),
                            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        )

        # Shortcut : UP[CBN에서 UP을 하냐 마냐 optional] - CONV or CONV
        self.short_cut = nn.Sequential(
                            nn.Upsample(scale_factor=2),
                            nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0)
        )      

    def forward(self, x, c):
        out = self.cbn_1(x, c)
        out = self.upconv(out)
        out = self.cbn_2(out, c)
        out = self.conv(out)
        
        return out + self.short_cut(x), c

class D_InitResBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(D_InitResBlock, self).__init__()
        
        self.block = nn.Sequential(
            spectral_norm(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(inplace=True),
            spectral_norm(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        
        # 여기는 왜 반대로 했지
        self.shortcut = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            spectral_norm(nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0))
        )
        
    def forward(self, x):
        return self.block(x) + self.shortcut(x)
        
class D_ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, downsampling=True):
        super(D_ResBlock, self).__init__()
        
        # Block : RELU - SNCONV - RELU - SNCONV - AVGPOOL(1/2)[optional]
        block_layers = []
        block_layers += [
            nn.ReLU(inplace=True),
            spectral_norm(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(inplace=True),
            spectral_norm(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1))
        ]
        if downsampling:
            block_layers += [nn.AvgPool2d(kernel_size=2, stride=2, padding=0)]
            
        self.block = nn.Sequential(* block_layers)
        
        # Shortcut : CONV(SN) - AVGPOOL(1/2)[optional]
        shorcut_layers = []
        shorcut_layers += [spectral_norm(nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0))]
        if downsampling:
            shorcut_layers += [nn.AvgPool2d(kernel_size=2, stride=2, padding=0)]
        
        self.shortcut = nn.Sequential(* shorcut_layers)
        
    def forward(self, x):
        return self.block(x) + self.shortcut(x)