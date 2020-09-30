# mnist 생성
import os
import math
from collections import OrderedDict

from network import net
from dataloader import data_loader
from metric import GAN_metric
#from util import loss_plot, metric_plot
from util import hinge_loss, byol_loss

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision

import numpy as np
import pickle

try:
    import nsml
except:
    print('Can not import NSML..!')
    
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

class Trainer():
    def __init__(self, config):
        print('\n Build BYOLDiscGAN trainer ...')
        self.config = config
        
        # Path
        output_path = os.path.join(config.output_root, config.model, config.dataset_name)
        self.loss_path = os.path.join(output_path, 'loss')
        self.img_path = os.path.join(output_path, 'img')
        self.weight_path = os.path.join(output_path, 'weight')
        self.metric_path = os.path.join(output_path, 'metric')
        
        # Hyperparameter
        self.batch_size = config.batch_size
        self.z_dim = config.z_dim
        self.n_dis_train = config.n_dis_train
        self.queue_size = config.queue_size
        self.m = config.m
        
        # Weights for loss
        self.lambda_adv = config.lambda_adv
        self.lambda_fm = config.lambda_fm
        self.lambda_byol = config.lambda_byol
        
        print('\n 1. Build the directories ... ')
        self._build_dirs()
        
        print('\n 2. Build a data loader ... ')
        self._build_dataloader()

        print('\n 3. Build the networks ...')
        self._build_network()

        print('\n 3. Build an evaluator (e.g., FID, IS calculator) ... ')
        self._build_evaluator()

        print('\n 4. Build a feature queue with the size of %d ... ' % config.queue_size)
        self._build_queue()
        
    def update_queue(self, new_img):
        '''
        Remove the old elements and replace them with the new elements from the queue
        '''
        
        # Update img
        img = self.queue['img']
        remaining_idx = img.size(0) - new_img.size(0)
        img = torch.cat([new_img.cpu(), img[:remaining_idx]], dim=0)
        self.queue['img'] = img
        
        # Update feature
        feature = self.queue['feature']
        new_feature = self.D.extract_feature(new_img).detach()
        new_feature = new_feature.view(new_feature.size(0), -1)
        
        feature = torch.cat([new_feature, feature[:remaining_idx]], dim=0)
        self.queue['feature'] = feature
        
        # Update used-or-not record
        used = self.queue['used']
        used = torch.cat([torch.zeros(new_img.size(0)).long(), used[:remaining_idx]], dim=0) 
        self.queue['used'] = used
        
    def update_target_network(self, m):
        D_param = self.D.named_parameters()
        target_net_param = self.target_net.named_parameters()

        target_net_param_dict = dict(target_net_param)
        
        # Copy the D's params to target_net's params
        for name, param in D_param:
            if name in target_net_param_dict:
                online_param = param.data
                target_param = target_net_param_dict[name].data
                new_target_param = target_param * m + online_param * (1. - m)
                target_net_param_dict[name].data.copy_(new_target_param)
            
    def slow_top_1_fm_loss(self, fake_img):
        '''
        Find the nearest real images with the given fake images and calculate the feature matching loss
        '''
        
        # fake_feature = [N, D] / real_featue = [Q, D]
        fake_feature = self.D.extract_feature(fake_img)
        N = fake_feature.size(0)
        fake_feature = fake_feature.view(N, -1)
        
        real_feature = self.queue['feature']
        
        feature_matching_loss = 0
        used_idx = []
        
        # 각 fake_img의 feature에 대해 nearest real_img feature를 찾고 둘 사이의 거리를 minimize
        for i in range(fake_feature.size(0)):
            fake_feature_i = fake_feature[i].unsqueeze(dim=0)
            dist = torch.sum((real_feature - fake_feature_i) ** 2, dim=1, keepdim=True)
            val, idx = torch.min(dist, dim=0)
                
            feature_matching_loss += val
            used_idx.append(idx)
            
        # calculate feature_matching_loss and reconrd the used indicies
        
        # 4x4 D feature 쓸 때
        #feature_matching_loss /= (N * C * H * W)
        
        # 1x1 D feature에 lambda 0.01
        feature_matching_loss /= N
        
        used_idx = torch.cat(used_idx, dim=0).long()
        self.queue['used'][used_idx] = 1
        return feature_matching_loss
    
    def train(self):
        epochs = 0
        total_iters = 0
        
        if self.config.resume:
            # weight, log, start_iter, epoch 로드
            _load_weight_and_log(self.config.start_iter)
            total_iters = self.config.start_iter
            epochs = int(total_iters * self.config.batch_size / self.dlen)

        print('Training start ...')
        while(True):
            for iters, (img, img_aug_1, img_aug_2) in enumerate(self.dloader):
                self._all_set_train()
                
                img = img.to(dev)
                
                if self.config.view_type == 'augmentation':
                    img_aug_1 = img_aug_1.to(dev)
                    img_aug_2 = img_aug_2.to(dev)
                elif self.config.view_type == 'shuffle':
                    img_aug_1 = img
                    img_aug_2 = img[torch.randperm(img.size(0))]
                elif self.config.view_type == 'half':
                    half = int(img.size(0) / 2)
                    img_aug_1 = img[:half]
                    img_aug_2 = img[half:]
                
                ''' ============================= Train D ============================= '''
                
                # D loss (1) - Adversarial loss
                D_real_score, img_recon = self.D(img, recon=True)

                z = torch.randn(img.size(0), self.z_dim).to(dev)
                fake_img = self.G(z).detach()
                D_fake_score = self.D(fake_img, recon=False)

                D_adv_loss = hinge_loss(D_real_score, 'D_real') + hinge_loss(D_fake_score, 'D_fake')
                
                # D loss (2) - Image recon loss
                D_recon_loss = torch.mean(torch.abs(img - img_recon))
                
                # D loss (3) - BYOL loss
                pred_1 = self.D.extract_pred_feature(img_aug_1)
                pred_2 = self.D.extract_pred_feature(img_aug_2)
                
                target_2 = self.target_net(img_aug_1)
                target_1 = self.target_net(img_aug_2)
                
                D_byol_loss = byol_loss(pred_1, target_1, pred_2, target_2)
                
                # D loss - Adversarial loss + BYOL loss
                D_loss = (self.config.lambda_adv * D_adv_loss) + \
                         (self.config.lambda_byol * D_byol_loss) + \
                         (self.config.lambda_recon * D_recon_loss)
                
                # Update D
                self._all_zero_grad()
                D_loss.backward()
                self.optim_D.step()
                
                # Update target_net
                self.update_target_network(m=self.config.m)
                
                ''' ============================= Train G ============================= '''
                
                if total_iters % self.config.n_dis_train == 0:
                    
                    # G loss (1) - Adversarial loss
                    z = torch.randn(img.size(0), self.config.z_dim).to(dev)
                    fake_img = self.G(z)
                    G_fake_score = self.D(fake_img, recon=False)

                    G_adv_loss = hinge_loss(G_fake_score, 'G')
                    
                    # G loss (2) - feature matching loss
                    if self.config.lambda_fm > 0:
                        feature_matching_loss = self.slow_top_1_fm_loss(fake_img)
                    else:
                        feature_matching_loss = torch.FloatTensor([0]).to(dev)
                    
                    G_loss = (self.config.lambda_adv * G_adv_loss) + \
                             (self.config.lambda_fm * feature_matching_loss)

                    # Update G
                    self._all_zero_grad()
                    G_loss.backward()
                    self.optim_G.step()
                    
                    # Save G_ema model
                    self._moving_average()

                # Update queue with the new real images
                self.update_queue(img)
                
                # Update total_iters 
                total_iters = total_iters + 1
                
                ''' ======================= B-3. Evaluation and log ======================== '''
                # Loss
                self._report_loss(epochs, total_iters, 
                                  D_adv_loss, D_recon_loss, D_byol_loss, G_adv_loss, feature_matching_loss)
                
                # Evaluate
                if (total_iters - 1) % self.config.save_img_iter == 0:
                    self._evaluate(total_iters)
                        
                # Save model
                if total_iters % self.config.save_weight_iter == 0:
                    self._save_weight(total_iters)
                    
            # 1 Epoch 완료
            epochs = epochs + 1

            
    ''' ================================= Utilities ================================= '''
    def _build_dirs(self):
        output_path = os.path.join(self.config.output_root, self.config.model, self.config.dataset_name)
        self.img_path = os.path.join(output_path, 'img')
        self.weight_path = os.path.join(output_path, 'weight')
        
        if not os.path.exists(self.img_path):
            os.makedirs(self.img_path)
        if not os.path.exists(self.weight_path):
            os.makedirs(self.weight_path)
            
    def _build_dataloader(self):
        self.dloader, self.dlen = data_loader(dataset_root=self.config.dataset_root, 
                                              dataset_name=self.config.dataset_name, 
                                              resize=self.config.resize, 
                                              batch_size=self.config.batch_size, 
                                              num_worker=self.config.num_worker,
                                              shuffle=True)
        
        self.queue_dloader, _ = data_loader(dataset_root=self.config.dataset_root, 
                                            dataset_name=self.config.dataset_name, 
                                            resize=self.config.resize, 
                                            batch_size=self.config.batch_size,
                                            num_worker=2,
                                            shuffle=False)
        
        
        
    def _build_network(self):
        if self.config.dataset_name == 'cifar-10' or self.config.dataset_name == 'cifar-10-npz':
            self.num_classes = 10
            self.D = net.Discriminator_32().to(dev)
            self.G = net.Generator_32(z_dim=self.config.z_dim).to(dev)
            self.G_ema = net.Generator_32(z_dim=self.config.z_dim).to(dev)            
        
        elif self.config.dataset_name == 'celeba_hq_images':
            self.D = net.BYOLReconDiscriminator_128(hidden_dim=self.config.hidden_dim, 
                                                    projection_dim=self.config.projection_dim,
                                                    recon_dim=self.config.recon_dim, 
                                                    feature_loc=self.config.feature_loc).to(dev)
            self.target_net = net.TargetNetwork_128(hidden_dim=self.config.hidden_dim, 
                                                    projection_dim=self.config.projection_dim).to(dev)
            self.G = net.Generator_128(z_dim=self.config.z_dim).to(dev)
            self.G_ema = net.Generator_128(z_dim=self.config.z_dim).to(dev)

            
        # Initialize target network
        self.update_target_network(m=0)
        for param in self.target_net.parameters():
            param.requires_grad = False
        
        # Build optimizer
        self.optim_D = optim.Adam(self.D.parameters(), 
                                  lr=self.config.D_lr, 
                                  betas=(self.config.beta_1, self.config.beta_2))
        self.optim_G = optim.Adam(self.G.parameters(), 
                                  lr=self.config.G_lr, 
                                  betas=(self.config.beta_1, self.config.beta_2))
        
    def _build_evaluator(self):
        # Load pre-calcualted stats
        if self.config.dataset_name == 'cifar-10' or self.config.dataset_name == 'cifar-10-npz':
            self.real_fid_stat_path = os.path.join(self.config.dataset_root, 
                                                   self.config.dataset_name, 
                                                   'fid_stats_cifar10_train.npz')
        elif self.config.dataset_name == 'celeba_hq_images':
            self.real_fid_stat_path = os.path.join(self.config.dataset_root, 
                                                   self.config.dataset_name, 
                                                   'fid_stats_celeba_hq_train_fixed.npz')
            self.real_ipr_stat_path = os.path.join(self.config.dataset_root, 
                                                   self.config.dataset_name, 
                                                   'IPR_CelebA_HQ.npz')
            
        # Build the evaluators
        if self.config.eval_FID:
            self.FID = GAN_metric.FID(dims=self.config.FID_dims, 
                                      batch_size=self.config.FID_batch_size, 
                                      real_stat_path=self.real_fid_stat_path)

        if self.config.eval_IS:
            self.IS = GAN_metric.InceptionScore(batch_size=self.config.IS_batch_size, 
                                                resize=True,
                                                splits=self.config.IS_splits, 
                                                renormalize=True)
        if self.config.eval_IPR:
            self.IPR = GAN_metric.IPR(batch_size=self.config.IPR_batch_size, 
                                      k=self.config.IPR_k,
                                      real_stat_path=self.real_ipr_stat_path)

        if self.config.eval_KNN:
            self.KNN = GAN_metric.KNN(knn_eval_k=self.config.knn_eval_k,
                                      save_path=self.img_path)
        
        # Fixed random vector z for evaluating
        self.fixed_z = torch.randn(self.config.gen_img_num, self.config.z_dim).to(dev)
            
            
    def _build_queue(self):
        queue_img = []
        queue_feature = []
        
        with torch.no_grad():
            for i, (img, _, _) in enumerate(self.dloader):
                feature = self.D.extract_feature(img.to(dev)).detach()
                feature = feature.view(feature.size(0), -1)
                
                queue_img.append(img)
                queue_feature.append(feature)

                if i == (self.config.queue_size / self.config.batch_size) - 1:
                    break
                    
        queue_img = torch.cat(queue_img, dim=0)
        queue_feature = torch.cat(queue_feature, dim=0)
        queue_used = torch.zeros(queue_img.size(0)).long()
        self.queue = {'img': queue_img, 'feature': queue_feature, 'used': queue_used}
        
    def _save_weight(self, iters):
        file_name = 'ckpt_' + str(iters) + '.pkl'
        path_ckpt = os.path.join(self.weight_path, file_name)
        ckpt = {
             'D': self.D.state_dict(),
             'target_net': self.target_net.state_dict(),
             'G': self.G.state_dict(),
             'G_ema': self.G_ema.state_dict(),
             'fixed_z': self.fixed_z
        }
        torch.save(ckpt, path_ckpt)
            
    def _load_weight(self, iters):
        file_name = 'ckpt_' + str(iters) + '.pkl'
        path_ckpt = os.path.join(self.weight_path, file_name)
        ckpt = torch.load(path_ckpt)
        
        self.D.load_state_dict(ckpt['D'])
        self.target_net.load_state_dict(ckpt['target_net'])
        self.G.load_state_dict(ckpt['G'])
        self.G_ema.load_state_dict(ckpt['G_ema'])
        
        self.fixed_z = ckpt['fixed_z']
        return iters
    
    def _all_set_train(self):
        self.D.train()
        self.target_net.train()
        self.G.train()

    def _all_set_eval(self):
        self.D.eval()
        self.target_net.eval()
        self.G.eval()

    def _all_zero_grad(self):
        self.optim_D.zero_grad()
        self.target_net.zero_grad()
        self.optim_G.zero_grad()
        
    def _lr_update(self):            
        self.scheduler_D.step()
        self.scheduler_G.step()
        
    def _moving_average(self, beta=0.999):
        for param, param_test in zip(self.G.parameters(), self.G_ema.parameters()):
            param_test.data = torch.lerp(param.data, param_test.data, beta)

    def _gen_eval_imgs_and_save(self, total_iters):
        output_img_path = os.path.join(self.img_path, str(total_iters - 1) + '.png')
        
        ##################################### Random generation #####################################
        with torch.no_grad():
            eval_img = []
            display_img = []
            
            # fixed_z로 생성된 이미지를 한번에 GPU에 올리면 터질 위험이 높음. 나눠서 생성.
            for i in range(int(self.config.gen_img_num / self.config.FID_batch_size)):
                start = i * self.config.FID_batch_size
                end = min((i + 1) * self.config.FID_batch_size, self.config.gen_img_num)
                fake_img = self.G(self.fixed_z[start:end]).cpu()
                eval_img.append(fake_img)

            # display 할 이미지의 수가 FID 측정용 이미지의 수보다 적을 경우 앞에서 잘라서 저장
            display_img = eval_img
            if self.config.display_img_num < self.config.gen_img_num:
                display_img = display_img[:int(self.config.display_img_num / 
                                               self.config.FID_batch_size)]

            # 이미지 저장
            display_img = torch.cat(display_img, dim=0)
            torchvision.utils.save_image(display_img, 
                                         output_img_path, 
                                         nrow=int(math.sqrt(self.config.display_img_num)),
                                         normalize=True,
                                         padding=0)

            # 이미지 생성 후 FID 측정
            eval_img = torch.cat(eval_img, dim=0)
            
        ##################################### Reconstruction #####################################
        output_img_path = os.path.join(self.img_path, 'recon_' + str(total_iters - 1) + '.png')
        
        with torch.no_grad():
            input_img_list = []
            recon_img_list = []
            input_img_feature_list = []
            
            for i, (img, _, _) in enumerate(self.queue_dloader):
                img = img.to(dev)
                _, recon_img = self.D(img, recon=True)
                feature = self.D.extract_feature(img)
                feature = feature.view(feature.size(0), -1)
                
                input_img_list.append(img.cpu())
                recon_img_list.append(recon_img.cpu())
                input_img_feature_list.append(feature)
                
                if (i + 1) * self.config.batch_size >= self.config.knn_key_img_num:
                    break
                
            input_img = torch.cat(input_img_list, dim=0)
            recon_img = torch.cat(recon_img_list, dim=0)
            input_img_feature = torch.cat(input_img_feature_list, dim=0)
                
            display_img = torch.cat([input_img.unsqueeze(dim=1)[:self.config.display_img_num], 
                                     recon_img.unsqueeze(dim=1)[:self.config.display_img_num]], dim=1)
            display_img = display_img.view(-1, 3, self.config.resize, self.config.resize)
            torchvision.utils.save_image(display_img, 
                                         output_img_path, 
                                         nrow=16,
                                         normalize=True,
                                         padding=0)
            
        return input_img, input_img_feature, eval_img
    
    def _report_loss(self, epochs, total_iters, D_adv_loss, D_recon_loss, D_byol_loss, 
                     G_adv_loss, feature_matching_loss):
        
        print('  ** [Epoch : %d / Total iter : %d] => D adv loss : %f / D recon loss : %f / D byol loss : %f / G adv loss : %f / FM loss : %f / Queue util : %f'% (epochs+1, total_iters, D_adv_loss.item(), D_recon_loss.item(), D_byol_loss.item(), G_adv_loss.item(), feature_matching_loss.item(), torch.mean(self.queue['used'].float())))
            
        log = OrderedDict()
        log['loss/D_adv'] = D_adv_loss.item()
        log['loss/D_recon'] = D_recon_loss.item()
        log['loss/D_byol'] = D_byol_loss.item()
        log['loss/G_adv'] = G_adv_loss.item()
        log['loss/FM'] = feature_matching_loss.item()
        log['queue_util'] = torch.mean(self.queue['used'].float()).item()
            
        if (total_iters - 1) % self.config.report_loss_iter == 0 and self.config.use_nsml:
            nsml.report(**log, scope=locals(), step=total_iters)
    
    def _evaluate(self, total_iters):        
        # Generate imgs and save
        self.G.eval()
        real_img, real_feature, eval_img = self._gen_eval_imgs_and_save(total_iters)

        log = OrderedDict()

        # Calculate FID
        if self.config.eval_FID:
            print('\n    ** [Evaluation]: calculate FID score ...')
            fid_score = self.FID.get_fid_score(eval_img)
            log['eval/FID'] = fid_score.item()
            
        # Calcualte IS
        if self.config.eval_IS:
            print('\n    ** [Evaluation]: Calculate Inception score ...')
            inception_score, _ = self.IS.get_inception_score(eval_img)
            log['eval/IS'] = inception_score.item()
            
        # KNN
        if self.config.eval_KNN:
            print('\n    ** [Evaluation]: KNN ...')
            used_num = self.KNN.get_knn(model=self.D,
                                        real_img=real_img, 
                                        real_feature=real_feature, 
                                        eval_img=eval_img[:self.config.knn_query_img_num], 
                                        total_iters=total_iters)
            log['eval/KNN_num'] = used_num
                
                
        # Calculate precision and recall
        if self.config.eval_IPR:
            print('\n    ** [Evaluation]: Precision and recall ...')
            precision, recall = self.IPR.get_precision_and_recall(eval_img)
            log['eval/Precision'] = precision
            log['eval/Reacll'] = recall

        # Report scores to nsml
        if self.config.use_nsml:
            nsml.report(**log, scope=locals(), step=total_iters)