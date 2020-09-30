import os
from network import net
from metric import GAN_metric
import torch

dev = 'cuda' if torch.cuda.is_available() else 'cpu'

weight_path = 'gen_result/KR62512_generation-nfs_551/app/expr/FMSNGAN/sanghyeon/cifar-10-npz/weight/ckpt_500000.pkl'
dims = 2048
batch_size = 50
real_stat_path = '/data_nfs/generation-nfs/sanghyeon/cifar-10-npz/fid_stats_cifar10_npz_train.npz'

z_dim = 128
gen_img_num = 10000

with torch.no_grad():
    print('Build generator')
    G = net.Generator_32(z_dim=z_dim).to(dev)

    ckpt = torch.load(weight_path)
    G.load_state_dict(ckpt['G'])

    print('Build evaluator')
    fid_calculator = GAN_metric.FID(dims, batch_size, real_stat_path)

    eval_img = []
    print('Generate images')
    for i in range(gen_img_num // batch_size):

        print('[%d / %d] ... ' % (i, gen_img_num // batch_size))

        z = torch.randn(batch_size, z_dim).to(dev)
        fake_img = G(z).cpu()
        eval_img.append(fake_img)

    print('Calculate FID score')
    eval_img = torch.cat(eval_img, dim=0)
    fid_score = fid_calculator.get_fid_score(eval_img)
    print(fid_score.item())