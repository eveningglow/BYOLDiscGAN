import os
from functools import partial
from collections import namedtuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import adaptive_avg_pool2d

from torchvision.models.inception import inception_v3
import torchvision.models as models
from torchvision import transforms
import torchvision

from . import inception

import numpy as np
from scipy import linalg
from scipy.stats import entropy


try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x): return x
    
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

class FID():
    def __init__(self, dims, batch_size, real_stat_path):
        block_idx = inception.InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        self.inception_net = inception.InceptionV3([block_idx], normalize_input=False).to(dev)
        self.inception_net.eval()
        
        self.batch_size = batch_size
        self.dims = dims
        
        f = np.load(real_stat_path)
        self.real_mu, self.real_sigma = f['mu'][:], f['sigma'][:]
        
    def get_activations(self, img):
        cur_batch_size = self.batch_size
        
        if img.size(0) % self.batch_size != 0:
            print(('Warning: number of images is not a multiple of the '
                   'batch size. Some samples are going to be ignored.'))
        if self.batch_size > img.size(0):
            print(('Warning: batch size is bigger than the data size. '
                   'Setting batch size to data size'))
            cur_batch_size = img.size(0)

        n_batches = img.size(0) // cur_batch_size
        n_used_imgs = n_batches * cur_batch_size

        pred_arr = np.empty((n_used_imgs, self.dims))

        for i in tqdm(range(n_batches)):
            start = i * cur_batch_size
            end = start + cur_batch_size

            # 싹 다 gpu에 넣으면 터지고 batch_size 만큼씩만 gpu에 넘겨준다.
            # batch = img[start:end].to(dev) / 2 + 0.5
            batch = img[start:end].to(dev)

            # model에 그대로 -1 ~ 1 넘겨주기
            # model은 내부에 norm 없음
            pred = self.inception_net(batch)[0]

            if pred.shape[2] != 1 or pred.shape[3] != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred_arr[start:end] = pred.cpu().data.numpy().reshape(cur_batch_size, -1)

        return pred_arr
    
    def calculate_stats(self, activations):
        return np.mean(activations, axis=0), np.cov(activations, rowvar=False)
    
    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)
    
    def get_fid_score(self, imgs):
        # gen_imgs : -1 ~ 1
        feature = self.get_activations(imgs)
        fake_mu, fake_sigma = self.calculate_stats(feature)
        return self.calculate_frechet_distance(self.real_mu, self.real_sigma, fake_mu, fake_sigma)

class InceptionScore():
    def __init__(self, batch_size=32, resize=True, splits=10, renormalize=True):
        # Load inception model
        self.inception_net = inception_v3(pretrained=True, transform_input=False).to(dev)
        self.inception_net.eval()
        
        self.batch_size = batch_size
        self.resize = resize
        self.splits = splits
        self.renormalize = True
        
        self.up = nn.Upsample(size=(299, 299)).to(dev)
    
    def get_pred(self, x):
        if self.resize:
            x = self.up(x)
        x = self.inception_net(x)
        return F.softmax(x, dim=1).data.cpu().numpy()
    
    def get_inception_score(self, imgs):
        # imgs : -1 ~ 1
        N = len(imgs)

        assert self.batch_size > 0
        assert N > self.batch_size

        # Renormalize tensor
        if self.renormalize:
            mean=[0.485, 0.456, 0.406]
            std=[0.229, 0.224, 0.225]
            mean = torch.FloatTensor(mean).unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3)
            std = torch.FloatTensor(std).unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3)
            imgs = imgs / 2 + 0.5
            imgs = (imgs - mean) / std

        # Get predictions
        preds = np.zeros((N, 1000))
        n_batches = int(imgs.size(0) / self.batch_size) + 1

        for i in tqdm(range(n_batches)):
            start = i * self.batch_size
            end = start + self.batch_size
            if end > imgs.size(0):
                end = imgs.size(0)

            batch = imgs[start:end].to(dev)
            batch_size_i = batch.size(0)
            start = i * self.batch_size
            end = i * self.batch_size + batch_size_i
            preds[start : end] = self.get_pred(batch)
            #preds[i * self.batch_size : i*self.batch_size + batch_size_i] = get_pred(batch)

        # Now compute the mean kl-div
        split_scores = []

        for k in range(self.splits):
            part = preds[k * (N // self.splits): (k+1) * (N // self.splits), :]
            
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))

        return np.mean(split_scores), np.std(split_scores)

Manifold = namedtuple('Manifold', ['features', 'radii'])

class IPR():
    def __init__(self, batch_size, k, real_stat_path):
#         self.manifold_ref = None
        self.manifold_ref = self.compute_manifold(real_stat_path)
        self.batch_size = batch_size
        self.k = k
        self.vgg16 = models.vgg16(pretrained=True).to(dev).eval()
        
    def get_precision_and_recall(self, fake_images):
        '''
        Compute precision and recall for given subject
        reference should be precomputed by IPR.compute_manifold_ref()
        args:
            subject: path or images
                path: a directory containing images or precalculated .npz file
                images: torch.Tensor of N x C x H x W
        returns:
            PrecisionAndRecall
        '''
        assert self.manifold_ref is not None, "call IPR.compute_manifold_ref() first"

        manifold_subject = self.compute_manifold(fake_images)
        precision = self.compute_metric(self.manifold_ref, manifold_subject.features, 'computing precision...')
        recall = self.compute_metric(manifold_subject, self.manifold_ref.features, 'computing recall...')
        return precision, recall

    def compute_manifold(self, input):
        # npz precalculated file
        if isinstance(input, str):
            if input.endswith('.npz'):  
                f = np.load(input)
                feats = f['feature']
                radii = f['radii']
                f.close()
#                 print(feats, radii)
                return Manifold(feats, radii)
        # Tensor
        elif isinstance(input, torch.Tensor):
            feats = self.extract_features(input)
        else:
            print(type(input))
            raise TypeError

        # radii
        distances = self.compute_pairwise_distances(feats)
        radii = self.distances2radii(distances, k=self.k)
        return Manifold(feats, radii)

    def extract_features(self, images):
        desc = 'extracting features of %d images' % images.size(0)
        num_batches = int(np.ceil(images.size(0) / self.batch_size))

        # images : -1 ~ 1 -> 0 ~ 1 (PIL range) -> resize -> imagenet norm / (128 x 128) -> (224, 224)
        mean = torch.FloatTensor([0.485,0.456,0.406]).unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3)
        std = torch.FloatTensor([0.229,0.224,0.225]).unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3)
    
        images = F.interpolate(images, size=(224, 224), mode='bilinear')
        images = images / 2 + 0.5
        images = (images - mean) / std

        features = []
        
        for bi in range(num_batches):
            start = bi * self.batch_size
            end = start + self.batch_size
            batch = images[start:end]
            before_fc = self.vgg16.features(batch.to(dev))
            before_fc = before_fc.view(-1, 7 * 7 * 512)
            feature = self.vgg16.classifier[:4](before_fc)
            features.append(feature.cpu().data.numpy())

        return np.concatenate(features, axis=0)


    def compute_pairwise_distances(self, X, Y=None):
        '''
        args:
            X: np.array of shape N x dim
            Y: np.array of shape N x dim
        returns:
            N x N symmetric np.array
        '''
        num_X = X.shape[0]
        if Y is None:
            num_Y = num_X
        else:
            num_Y = Y.shape[0]
        X = X.astype(np.float64)  # to prevent underflow
        X_norm_square = np.sum(X**2, axis=1, keepdims=True)
        if Y is None:
            Y_norm_square = X_norm_square
        else:
            Y_norm_square = np.sum(Y**2, axis=1, keepdims=True)
        X_square = np.repeat(X_norm_square, num_Y, axis=1)
        Y_square = np.repeat(Y_norm_square.T, num_X, axis=0)
        if Y is None:
            Y = X
        XY = np.dot(X, Y.T)
        diff_square = X_square - 2*XY + Y_square

        # check negative distance
        min_diff_square = diff_square.min()
        if min_diff_square < 0:
            idx = diff_square < 0
            diff_square[idx] = 0
            print('WARNING: %d negative diff_squares found and set to zero, min_diff_square=' % idx.sum(),
                  min_diff_square)

        distances = np.sqrt(diff_square)
        return distances

    def distances2radii(self, distances, k=3):
        num_features = distances.shape[0]
        radii = np.zeros(num_features)
        for i in range(num_features):
            radii[i] = self.get_kth_value(distances[i], k=k)
        return radii

    def get_kth_value(self, np_array, k):
        kprime = k+1  # kth NN should be (k+1)th because closest one is itself
        idx = np.argpartition(np_array, kprime)
        k_smallests = np_array[idx[:kprime]]
        kth_value = k_smallests.max()
        return kth_value

    def compute_metric(self, manifold_ref, feats_subject, desc=''):
        num_subjects = feats_subject.shape[0]
        count = 0
        dist = self.compute_pairwise_distances(manifold_ref.features, feats_subject)
        for i in range(num_subjects):
            count += (dist[:, i] < manifold_ref.radii).any()
        return count / num_subjects
        
# class KNN():
#     def __init__(self, model, knn_eval_k, save_path):
#         self.model = model
#         self.knn_eval_k = knn_eval_k
#         self.save_path = save_path
        
#     def get_knn(self, real_img, real_feature, eval_img, total_iters):
#         with torch.no_grad():
#             knn_result_img = []
#             used_idx = []

#             # Get top-k real image from the queue for each fake image
#             for i in range(eval_img.size(0)):
#                 eval_img_i = eval_img[i].unsqueeze(dim=0).to(dev)
#                 eval_feature_i = self.model.extract_feature(eval_img_i)
#                 eval_feature_i = eval_feature_i.view(eval_feature_i.size(0), -1)

#                 dist = torch.sum((real_feature - eval_feature_i) ** 2, dim=1)
#                 val, idx = torch.topk(dist, self.knn_eval_k, dim=0, largest=False, sorted=True)

#                 top_k_img = real_img[idx]

#                 knn_result_img.append(eval_img_i.cpu())
#                 knn_result_img.append(top_k_img)
                
#                 # Get top-k idx
#                 used_idx.append(idx)

#             # 이미지 저장
#             knn_img_path = os.path.join(self.save_path, 'knn_' + str(total_iters - 1) + '.png')
#             knn_result_img = torch.cat(knn_result_img, dim=0) / 2 + 0.5
#             torchvision.utils.save_image(knn_result_img, 
#                                          knn_img_path, 
#                                          nrow=self.knn_eval_k+1,
#                                          normalize=False,
#                                          padding=0)
#             print('    ** [Evaluation]: Save ' + knn_img_path)
            
#             # Remove duplicates and count
#             used_num = torch.unique(torch.cat(used_idx, dim=0)).cpu().size(0)
#             return used_num
        
class KNN():
    def __init__(self, knn_eval_k, save_path):
        self.knn_eval_k = knn_eval_k
        self.save_path = save_path
        
    def get_knn(self, model, real_img, real_feature, eval_img, total_iters):
        with torch.no_grad():
            knn_result_img = []
            used_idx = []

            # Get top-k real image from the queue for each fake image
            for i in range(eval_img.size(0)):
                eval_img_i = eval_img[i].unsqueeze(dim=0).to(dev)
                eval_feature_i = model.extract_feature(eval_img_i)
                eval_feature_i = eval_feature_i.view(eval_feature_i.size(0), -1)
                
                dist = torch.sum((real_feature - eval_feature_i) ** 2, dim=1)
                val, idx = torch.topk(dist, self.knn_eval_k, dim=0, largest=False, sorted=True)

                top_k_img = real_img[idx]

                knn_result_img.append(eval_img_i.cpu())
                knn_result_img.append(top_k_img)
                
                # Get top-k idx
                used_idx.append(idx)

            # 이미지 저장
            knn_img_path = os.path.join(self.save_path, 'knn_' + str(total_iters - 1) + '.png')
            knn_result_img = torch.cat(knn_result_img, dim=0) / 2 + 0.5
            torchvision.utils.save_image(knn_result_img, 
                                         knn_img_path, 
                                         nrow=self.knn_eval_k+1,
                                         normalize=False,
                                         padding=0)
            print('    ** [Evaluation]: Save ' + knn_img_path)
            
            # Remove duplicates and count
            used_num = torch.unique(torch.cat(used_idx, dim=0)).cpu().size(0)
            return used_num