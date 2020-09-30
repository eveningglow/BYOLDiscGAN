import os
import argparse
from torch.backends import cudnn
import torch
from distutils.util import strtobool
import pprint

def print_config(config):
    print(' \n================================== CONFIG ==================================\n')
    pprint.pprint(vars(config))
    print(' \n============================================================================\n')
    
    
def main(config):
    print_config(config)
    
    cudnn.benchmark = True
    torch.manual_seed(config.random_seed)
    
    try:
        import nsml
        config.dataset_root = nsml.DATASET_PATH
    except:
        config.dataset_root = '/data_nfs/camist003/user/yunjey_nfs/'
        print('\nCan not import NSML...!')
        
    # Start training
    print('\nStart Training ...')
    
    if config.model == 'BYOLDiscGAN':
        from model import BYOLDiscGAN
        trainer = BYOLDiscGAN.Trainer(config)
    elif config.model == 'CRGAN':
        from model import CRGAN
        trainer = CRGAN.Trainer(config)
    elif config.model == 'BYOLReconDiscGAN':
        from model import BYOLReconDiscGAN
        trainer = BYOLReconDiscGAN.Trainer(config)

    trainer.train()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Config - model
    parser.add_argument('--random_seed', type=int, default=666)
    parser.add_argument('--model', type=str, default='BYOLDiscGAN')
    
    # Config - path
    parser.add_argument('--dataset_root', type=str, default='/data_nfs/camist003/user/yunjey_nfs/')
    parser.add_argument('--dataset_name', type=str, default='celeba_hq_images')
    parser.add_argument('--output_root', type=str, default='expr')
    
    # Config - img preprocess
    parser.add_argument('--resize', type=int, default=128)
    
    # Config - optimizer
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--D_lr', type=float, default=0.0002)
    parser.add_argument('--G_lr', type=float, default=0.0002)
    parser.add_argument('--beta_1', type=float, default=0.0)
    parser.add_argument('--beta_2', type=float, default=0.9)
    parser.add_argument('--lr_decay', action='store_true')
    parser.add_argument('--m', type=float, default=0.996)
    
    # Config - architectural setting
    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--feature_loc', type=int, default=-1)
    parser.add_argument('--queue_size', type=int, default=2048)        
    parser.add_argument('--hidden_dim', type=int, default=2048)
    parser.add_argument('--projection_dim', type=int, default=128)
    parser.add_argument('--recon_dim', type=int, default=128)
    
    # Config - other hyperparameters
    parser.add_argument('--n_dis_train', type=int, default=3)
    parser.add_argument('--view_type', type=str, default='half')
    
    # Config - loss type
    parser.add_argument('--cr_type', type=str, default='logit')
    
    # Config - lambda for each loss
    parser.add_argument('--lambda_adv', type=float, default=1)
    parser.add_argument('--lambda_fm', type=float, default=0)
    parser.add_argument('--lambda_byol', type=float, default=1)
    parser.add_argument('--lambda_recon', type=float, default=1)
    parser.add_argument('--lambda_cr', type=float, default=0)
    
    # Config - Log
    parser.add_argument('--gen_img_num', type=int, default=10000)
    parser.add_argument('--display_img_num', type=int, default=100)
    parser.add_argument('--save_img_iter', type=int, default=10000)
    parser.add_argument('--save_weight_iter', type=int, default=50000)
    parser.add_argument('--report_loss_iter', type=int, default=100)
    parser.add_argument('--num_worker', type=int, default=8)
    
    # Config - FID
    parser.add_argument('--eval_FID', type=strtobool, default=True)
    parser.add_argument('--FID_batch_size', type=int, default=50)
    parser.add_argument('--FID_dims', type=int, default=2048)
    
    # Config - IS
    parser.add_argument('--eval_IS', type=strtobool, default=False)
    parser.add_argument('--IS_batch_size', type=int, default=32)
    parser.add_argument('--IS_splits', type=int, default=1)
    parser.add_argument('--IS_renormalize', action='store_true')
    
    # Config - KNN
    parser.add_argument('--eval_KNN', type=strtobool, default=True)
    parser.add_argument('--knn_query_img_num', type=int, default=64)
    parser.add_argument('--knn_key_img_num', type=int, default=1024)
    parser.add_argument('--knn_eval_k', type=int, default=10)
    
    # Config - Precision and recall
    parser.add_argument('--eval_IPR', type=strtobool, default=False)
    parser.add_argument('--IPR_batch_size', type=int, default=50)
    parser.add_argument('--IPR_k', type=int, default=3)
    
    # Config - extra
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--use_nsml', type=strtobool, default=True)
    

    config = parser.parse_args()
    main(config)