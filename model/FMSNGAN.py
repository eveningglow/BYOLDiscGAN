# mnist 생성
import os
import math
from collections import OrderedDict

from network import net
from dataloader import data_loader
from metric import GAN_metric
#from util import loss_plot, metric_plot
from util import HingeLoss

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
        print('Build FMSNGAN trainer ...')
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
        
        # Weights for loss
        self.lambda_adv = config.lambda_adv
        self.lambda_fm = config.lambda_fm
        
        print('\n 1. Build a data loader ... ')
        self._build_dataloader(config)

        print('\n 2. Build networks ...')
        self._build_network(config)

        print('\n 3. Build an evaluator (e.g., FID, IS calculator) ... ')
        self._build_evaluator(config)

        print('\n 4. Build a feature queue with the size of %d ... ' % config.queue_size)
        self._build_queue(config)

        print('\n 5. Build loggers ... ')
        self._build_logger(config)
        
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
        
    def moving_average(self, beta=0.999):
        for param, param_test in zip(self.G.parameters(), self.G_ema.parameters()):
            param_test.data = torch.lerp(param.data, param_test.data, beta)
            
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
            # fake_feature_i = [1, D]
            fake_feature_i = fake_feature[i].unsqueeze(dim=0)
            
            # ([Q, D] - [1, D]) -> [Q, D] -> [Q, 1]
            #dist = torch.sum(torch.abs(real_feature - fake_feature_i), dim=1, keepdim=True)
            
            # 4x4 D feature 쓸 때 / 1x1 D feature에 lambda 0.01
            dist = torch.sum((real_feature - fake_feature_i) ** 2, dim=1, keepdim=True)
            
            # 각 fake_img 마다 nearest real_img feature와의 거리를 찾고 해당 idx 저장
            val, idx = torch.min(dist, dim=0)
            
            # Nearest는 깐깐하게, constraint는 유하게
            if self.weak_fm_loss:
                real_feature_i = F.normalize(real_feature[idx], dim=1)
                fake_feature_i = F.normalize(fake_feature_i, dim=1)
                val = -torch.sum(real_feature_i * fake_feature_i, dim=1)
                
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
            
    def top_1_fm_loss(self, fake_img):
        '''
        Find the nearest real images with the given fake images and calculate the feature matching loss
        Not fully updated yet
        '''
        
        # fake_feature = [N, D]
        fake_feature = self.D.extract_feature(fake_img)
        feature_dim = fake_feature.size(1)
        
        # fake_feature = [N, D * Q] -> [N * Q, D] (dim=1 방향으로 복붙하고, dim=0으로 밀어넣기)
        fake_feature = fake_feature.repeat(1, self.queue_size).view(-1, feature_dim)

        # real_feature = [Q, D]
        real_feature = self.queue['feature']
        
        # real_feature = [Q * N, D]
        real_feature = real_feature.repeat(fake_img.size(0), 1)

        # dist = [N * Q, 1] / 모든 N개의 fake_img와 모든 Q개의 queue_img 사이의 feature distance
        dist = torch.sum(torch.abs(real_feature - fake_feature), dim=1, keepdim=True)
        
        # 각 dist_chunk = [Q, 1] / 총 N개의 dist_chunk / 각 dist_chunk는 각 fake_img와 모든 queue_img 사이의 distance
        dist_chunk = torch.split(dist, self.queue_size, dim=0)

        # dist_chunk = [Q, N] / dist_chunk의 각 column [Q, 1] 에는 각 fake_img와 모든 queue_img 사이의 distance
        dist_chunk = torch.cat(dist_chunk, dim=1)
        
        # val = [1, N] / used_idx = [N] / val의 각 element는 각 fake_img와 그의 nearest queue_img 사이의 distance
        # used_idx의 각 element는 각 fake_img의 nearest queue_img의 idx
        val, used_idx = torch.min(dist_chunk, dim=0)
        
        # Calculate feature_matching_loss and record the used indicies
        feature_matching_loss = torch.mean(val)
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

        print('\nTraining start ...')
        while(True):
            for iters, img in enumerate(self.dloader):
                self._all_set_train()
                
                img = img.to(dev)
                
                ''' ============================= Train D ============================= '''
                
                # Real image score to train D
                D_real_score = self.D(img)

                # Fake image score to train D
                z = torch.randn(img.size(0), self.z_dim).to(dev)
                fake_img = self.G(z).detach()
                D_fake_score = self.D(fake_img)

                # Calculate loss
                D_adv_loss = HingeLoss(D_real_score, 'D_real') + HingeLoss(D_fake_score, 'D_fake')
                D_adv_loss = self.lambda_adv * D_adv_loss

                # Update D
                self._all_zero_grad()
                D_adv_loss.backward()
                self.optim_D.step()
                
                ''' ============================= Train G ============================= '''
                
                if total_iters % self.n_dis_train == 0:
                    
                    # Fake image score to train G
                    z = torch.randn(img.size(0), self.z_dim).to(dev)
                    fake_img = self.G(z)
                    G_fake_score = self.D(fake_img)

                    # Calculate loss - adversarial loss
                    G_adv_loss = HingeLoss(G_fake_score, 'G')
                    G_adv_loss = self.lambda_adv * G_adv_loss
                    
                    # Calculate loss - feature matching loss
                    if self.lambda_fm > 0:
                        feature_matching_loss = self.slow_top_1_fm_loss(fake_img)
                    else:
                        feature_matching_loss = torch.FloatTensor([0]).to(dev)
                        
                    G_loss = G_adv_loss + (self.lambda_fm * feature_matching_loss)

                    # Update
                    self._all_zero_grad()
                    G_loss.backward()
                    self.optim_G.step()
                    
                    # Save G_ema model
                    self.moving_average()

                # Update queue with the new real images
                self.update_queue(img)
                
                # Update total_iters 
                total_iters = total_iters + 1
                
                ''' ============================= Evaluation and log ============================= '''
                
                print('  ** [Epoch : %d / Total iter : %d] => D adv loss : %f / G adv loss : %f / FM loss : %f / Queue util : %f'% (epochs+1, total_iters, D_adv_loss.item(), G_adv_loss.item(), feature_matching_loss.item(), torch.mean(self.queue['used'].float())))

                # Report losses
                if (total_iters - 1) % self.config.report_loss_iter == 0 and self.config.use_nsml:
                    log = OrderedDict()
                    log['loss/D_adv'] = D_adv_loss.item()
                    log['loss/G_adv'] = G_adv_loss.item()
                    log['loss/FM'] = feature_matching_loss.item()
                    log['queue_util'] = torch.mean(self.queue['used'].float()).item()
                    nsml.report(**log, scope=locals(), step=total_iters)
                
                if (total_iters - 1) % self.config.save_img_iter == 0:
                    self.G.eval()
                    eval_img = self._gen_eval_imgs_and_save(total_iters)

                    log = OrderedDict()
                    
                    # Calculate FID
                    if self.config.eval_FID:
                        print('\n  Evaluation: calculate FID score ...')
                        fid_score = self.FID.get_fid_score(eval_img)
                        log['eval/FID'] = fid_score.item()

                    # Calcualte IS
                    if self.config.eval_IS:
                        print('\n  Evaluation: Calculate Inception score ...')
                        inception_score, _ = self.IS.get_inception_score(eval_img)
                        log['eval/IS'] = inception_score.item()
                        
                    # KNN
                    if self.config.eval_KNN:
                        print('\n  Evaluation: KNN ...')
                        used_num = self._knn_test(eval_img, total_iters)
                        log['eval/KNN_num'] = used_num
                        
                    # Calculate precision and recall
                    if self.config.eval_IPR:
                        print('\n  Evaluation: Precision and recall ...')
                        precision, recall = self.IPR.get_precision_and_recall(eval_img)
                        log['eval/Precision'] = precision
                        log['eval/Reacll'] = recall
                        print(precision, recall)
                        
                    # Report to nsml
                    if self.config.use_nsml:
                        nsml.report(**log, scope=locals(), step=total_iters)
                        
                # Save model
                if total_iters % self.config.save_weight_iter == 0:
                    self._save_weight_and_log(total_iters)
                    
            # 1 Epoch 완료
            epochs = epochs + 1

            
    ''' ================================= Utilities ================================= '''
            
    def _build_dataloader(self):
        self.dloader, self.dlen = data_loader(dataset_root=self.config.dataset_root, 
                                              dataset_name=self.config.dataset_name, 
                                              resize=self.config.resize,
                                              crop=self.config.crop,
                                              batch_size=self.config.batch_size,
                                              num_worker=self.config.num_worker,
                                              shuffle=True)
        
        self.knn_dloader, _ = data_loader(dataset_root=self.config.dataset_root, 
                                          dataset_name=self.config.dataset_name, 
                                          resize=self.config.resize,
                                          crop=self.config.crop,
                                          batch_size=self.config.knn_key_img_num,
                                          num_worker=self.config.num_worker,
                                          shuffle=False)
        
    def _build_network(self):
        if self.config.dataset_name == 'cifar-10' or self.config.dataset_name == 'cifar-10-npz':
            self.num_classes = 10
            self.D = net.Discriminator_32().to(dev)
            self.G = net.Generator_32(z_dim=self.config.z_dim).to(dev)
            self.G_ema = net.Generator_32(z_dim=self.config.z_dim).to(dev)            
            self.real_fid_stat_path = os.path.join(self.config.dataset_root, self.config.dataset_name, 
                                                   'fid_stats_cifar10_train.npz')
        elif dataset_type == 'celeba_hq_images':
            self.D = net.Discriminator_128().to(dev)
            self.G = net.Generator_128(z_dim=self.config.z_dim).to(dev)
            self.G_ema = net.Generator_128(z_dim=self.config.z_dim).to(dev)
            self.real_fid_stat_path = os.path.join(self.config.dataset_root, self.config.dataset_name, 
                                                   'fid_stats_celeba_hq_train.npz')
            self.real_ipr_stat_path = os.path.join(self.config.dataset_root, self.config.dataset_name, 
                                                   'IPR_CelebA_HQ.npz')
            
        self.optim_D = optim.Adam(self.D.parameters(), 
                                  lr=self.config.D_lr, 
                                  betas=(self.config.beta_1, self.config.beta_2))
        
        self.optim_G = optim.Adam(self.G.parameters(), 
                                  lr=self.config.G_lr, 
                                  betas=(self.config.beta_1, self.config.beta_2))
        
    def _build_evaluator(self):
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
            
            
    def _build_queue(self):
        queue_img = []
        queue_feature = []
        
        with torch.no_grad():
            for i, img in enumerate(self.dloader):
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
        
    def _build_logger(self):
        if not os.path.exists(self.loss_path):
            os.makedirs(self.loss_path)
        if not os.path.exists(self.img_path):
            os.makedirs(self.img_path)
        if not os.path.exists(self.weight_path):
            os.makedirs(self.weight_path)
        if not os.path.exists(self.metric_path):
            os.makedirs(self.metric_path)
            
        self.loss_hist = {}
        self.loss_hist['D_adv'] = []
        self.loss_hist['G_adv'] = []
        self.loss_hist['FM'] = []
        
        self.FID_hist = {}
        self.FID_hist['FID'] = []
        
        self.IS_hist = {}
        self.IS_hist['IS'] = []
        
        self.fixed_z = torch.randn(self.config.gen_img_num, self.config.z_dim).to(dev)
        
    def _save_weight_and_log(self, iters):
        file_name = 'ckpt_' + str(iters) + '.pkl'
        path_ckpt = os.path.join(self.weight_path, file_name)
        ckpt = {
             'D': self.D.state_dict(),
             'G': self.G.state_dict(),
             'G_ema': self.G_ema.state_dict(),
             'loss_hist': self.loss_hist,
             'FID_hist': self.FID_hist,
             'IS_hist': self.IS_hist,
             'fixed_z': self.fixed_z
        }
        torch.save(ckpt, path_ckpt)
            
    def _load_weight_and_log(self, iters):
        file_name = 'ckpt_' + str(iters) + '.pkl'
        path_ckpt = os.path.join(self.weight_path, file_name)
        ckpt = torch.load(path_ckpt)
        
        self.D.load_state_dict(ckpt['D'])
        self.G.load_state_dict(ckpt['G'])
        self.G_ema.load_state_dict(ckpt['G_ema'])

        last_idx = iters // self.save_img_iter
        
        self.loss_hist = ckpt['loss_hist']
        self.loss_hist['D_adv'] = self.loss_hist['D_adv'][:last_idx]
        self.loss_hist['G_adv'] = self.loss_hist['G_adv'][:last_idx]
        self.loss_hist['FM'] = self.loss_hist['FM'][:last_idx]
        
        self.FID_hist = ckpt['FID_hist']
        self.FID_hist['FID'] = self.FID_hist['FID'][:last_idx]
        
        self.IS_hist = ckpt['IS_hist']
        self.IS_hist['IS'] = self.IS_hist['IS'][:last_idx]
        
        self.fixed_z = ckpt['fixed_z']
        return iters
    
    def _all_set_train(self):
        self.D.train()
        self.G.train()

    def _all_set_eval(self):
        self.D.eval()
        self.G.eval()

    def _all_zero_grad(self):
        self.optim_D.zero_grad()
        self.optim_G.zero_grad()
        
    def _lr_update(self):            
        self.scheduler_D.step()
        self.scheduler_G.step()
        
    def _gen_eval_imgs_and_save(self, total_iters):
        output_img_path = os.path.join(self.img_path, str(total_iters - 1) + '.png')
        
        with torch.no_grad():
            eval_img = []
            display_img = []
            
            # fixed_z를 한번에 GPU에 올리면 터질 위험이 높음. 나눠서 생성.
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
        return eval_img
    
    def _knn_test(self, eval_img, total_iters):
        with torch.no_grad():
            knn_result_img = []
            used_idx = []

            real_img_loader = iter(self.knn_dloader)
            
            real_img = real_img_loader.next()
            real_feature = self.D.extract_feature(real_img.to(dev))

            fake_img = eval_img[:self.config.knn_query_img_num]

            # Get top-k real image from the queue for each fake image
            for i in range(fake_img.size(0)):
                fake_img_i = fake_img[i].unsqueeze(dim=0).to(dev)
                fake_feature_i = self.D.extract_feature(fake_img_i)
                fake_feature_i = fake_feature_i.view(fake_feature_i.size(0), -1)

                dist = torch.sum((real_feature - fake_feature_i) ** 2, dim=1)
                val, idx = torch.topk(dist, self.config.knn_eval_k, dim=0, largest=False, sorted=True)

                top_k_img = real_img[idx]

                knn_result_img.append(fake_img_i.cpu())
                knn_result_img.append(top_k_img)
                
                # Get top-1 idx
                used_idx.append(idx)

            # 이미지 저장
            knn_img_path = os.path.join(self.img_path, 'knn_' + str(total_iters - 1) + '.png')
            knn_result_img = torch.cat(knn_result_img, dim=0) / 2 + 0.5
            torchvision.utils.save_image(knn_result_img, 
                                         knn_img_path, 
                                         nrow=self.config.knn_eval_k+1,
                                         normalize=False,
                                         padding=0)
            print('Save ' + knn_img_path)
            
            # Remove duplicates and count
            used_num = torch.unique(torch.cat(used_idx, dim=0)).cpu().size(0)
            return used_num