import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import warnings
from methods.base import ClassificationBaseModel


class CompressionNetForKNN(pl.LightningModule):
    def __init__(self, conv_module, X_fit, y_fit, n_compressed, n_neighbors, n_class, cfg_optimizer):
        super().__init__()
        self.conv_module = conv_module
        # (1, n_compressed)
        # self.compressed_y_fit = nn.Parameter(torch.Tensor((y_fit.reshape(-1,1))[:n_compressed])) # (n_compressed,1)
        self.register_buffer('compressed_y_fit', torch.Tensor((y_fit.reshape(-1,1))[:n_compressed]))
        self.register_buffer('X_fit', X_fit)
        self.n_neighbors = n_neighbors
        self.n_class = n_class
        self.cfg_optimizer = cfg_optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.grads = {}
        self.monitor_grad = False

    def forward(self):
        compressed_X_fit = (self.conv_module(self.X_fit)).cpu().detach().numpy()
        # eps = 1e-8
        # y_min = torch.min(self.compressed_y_fit)
        # y_max = torch.max(self.compressed_y_fit)
        # norm_compressed_y_fit = (self.compressed_y_fit - y_min)  / (y_max - y_min)
        # norm_compressed_y_fit = (self.n_class * norm_compressed_y_fit + eps)
        # ceil_compressed_y_fit = CeilNoGradient.apply(norm_compressed_y_fit)
        # compressed_y_fit = ceil_compressed_y_fit.cpu().detach().numpy().reshape(-1,)
        compressed_y_fit = (self.compressed_y_fit + 1).cpu().detach().numpy().reshape(-1,)

        results = {'compressed_X_fit':compressed_X_fit, 'compressed_y_fit':compressed_y_fit-1}
        # print(sorted(Counter(results['compressed_y_fit']).items()))
        return results
    
    def training_step(self, batch, batch_idx): 
        X, y = batch
        compressed_X_fit = self.conv_module(self.X_fit)
        dists = torch.cdist(compressed_X_fit, X).T #(batch_size, n_compressed)

        # generate mask
        mask = torch.zeros_like(dists).type_as(dists) # 用于存储每个test ins 的最
        dists = -(dists - torch.max(dists, dim=1, keepdim=True)[0]) #保证非负，将最小变为最大
        expand = 1e2
        for i in range(self.n_neighbors):
            temp_dist = dists - mask * dists # 将已记录的最大数变为0
            # softmax
            temp_dist = temp_dist - torch.max(temp_dist, dim=1, keepdim=True)[0] # 最大值为0
            temp_dist = temp_dist * expand
            temp_mask = F.softmax(temp_dist, dim=1) # 构造将最大值设为1的onehot向量
            mask = mask + temp_mask # 每轮每行都有一个值设为1
        
        eps = 1e-8
        # nomalize to [1, n_class], value is integer
        # y_min = torch.min(self.compressed_y_fit)
        # y_max = torch.max(self.compressed_y_fit)
        # norm_compressed_y_fit = (self.compressed_y_fit - y_min)  / (y_max - y_min)
        # norm_compressed_y_fit = (self.n_class * norm_compressed_y_fit + eps)
        # ceil_compressed_y_fit = CeilNoGradient.apply(norm_compressed_y_fit)
        ceil_compressed_y_fit = self.compressed_y_fit + 1

        # ceil_compressed_y_fit = norm_compressed_y_fit.detach().ceil()
        # floor_compre
        # norm_compressed_y_fit = norm_compressed_y_fit + ceil_compressed_y_fit
        
    
        # get y of top-min k dist
        y_mask = ceil_compressed_y_fit.T * mask #(batch_size, n_compressed)
        y_top_k = torch.topk(y_mask, k=self.n_neighbors, dim=1)[0]
        y_top_k = torch.unsqueeze(y_top_k, dim=-1) #升维 (batch, k, 1)

        id1 = y_top_k.detach()
        id2 = id1 - 1

        #(batch, k, n_class) one-hot
        onehot_top_k = torch.zeros(y_top_k.size(0), y_top_k.size(1), self.n_class).type_as(y_top_k).scatter(-1, id2.long(), 1)
        W = (y_top_k - id2).expand_as(onehot_top_k) #全为1的系数矩阵
        onehot_top_k = W*onehot_top_k # 插值使用变量索引变量，梯度能传递

        # 计算概率
        sum_top_k = torch.sum(onehot_top_k, dim=1) # (batch, n_class), 相当于logit
        # prob_top_k = F.softmax(sum_top_k, dim=1) 
        loss = self.criterion(sum_top_k, y.long())
        self.log('train_loss', loss)

        # register hook
        if self.monitor_grad:
            compressed_X_fit.register_hook(save_grad('compressed_X_fit',self.grads))
            norm_compressed_y_fit.register_hook(save_grad('norm compressed_y_fit',self.grads))
            ceil_compressed_y_fit.register_hook(save_grad('ceil compressed_y_fit',self.grads))
        return loss
    
    def validation_step(self, batch, batch_idx): 
        X, y = batch
        compressed_X_fit = self.conv_module(self.X_fit)
        dists = torch.cdist(compressed_X_fit, X).T #(batch_size, n_compressed)

        # generate mask
        mask = torch.zeros_like(dists).type_as(dists) # 用于存储每个test ins 的最
        dists = -(dists - torch.max(dists, dim=1, keepdim=True)[0]) #保证非负，将最小变为最大
        expand = 1e2
        for i in range(self.n_neighbors):
            temp_dist = dists - mask * dists # 将已记录的最大数变为0
            # softmax
            temp_dist = temp_dist - torch.max(temp_dist, dim=1, keepdim=True)[0] # 最大值为0
            temp_dist = temp_dist * expand
            temp_mask = F.softmax(temp_dist, dim=1) # 构造将最大值设为1的onehot向量
            mask = mask + temp_mask # 每轮每行都有一个值设为1
        
        eps = 1e-8
        # nomalize to [1, n_class], value is integer
        # y_min = torch.min(self.compressed_y_fit)
        # y_max = torch.max(self.compressed_y_fit)
        # norm_compressed_y_fit = (self.compressed_y_fit - y_min)  / (y_max - y_min)
        # norm_compressed_y_fit = (self.n_class * norm_compressed_y_fit + eps)
        # ceil_compressed_y_fit = CeilNoGradient.apply(norm_compressed_y_fit)
        ceil_compressed_y_fit = self.compressed_y_fit + 1
    
        # get y of top-min k dist
        y_mask = ceil_compressed_y_fit.T * mask #(batch_size, n_compressed)
        y_top_k = torch.topk(y_mask, k=self.n_neighbors, dim=1)[0]
        y_top_k = torch.unsqueeze(y_top_k, dim=-1) #升维 (batch, k, 1)

        id1 = y_top_k.detach()
        id2 = id1 - 1

        #(batch, k, n_class) one-hot
        onehot_top_k = torch.zeros(y_top_k.size(0), y_top_k.size(1), self.n_class).type_as(y_top_k).scatter(-1, id2.long(), 1)
        W = (y_top_k - id2).expand_as(onehot_top_k) #全为1的系数矩阵
        onehot_top_k = W*onehot_top_k # 插值使梯度能传递

        # 计算概率
        sum_top_k = torch.sum(onehot_top_k, dim=1) # (batch, n_class), 相当于logit
        # prob_top_k = F.softmax(sum_top_k, dim=1) 
        loss = self.criterion(sum_top_k, y.long())
        self.log('train_loss', loss)
    
    def configure_optimizers(self):
        if self.cfg_optimizer.NAME.lower() == 'adam':
            return torch.optim.Adam(self.parameters(), lr=self.cfg_optimizer.BASE_LR, betas=(0.9, 0.999), eps=1e-08)
        else:
            warnings.warn('(compression_net.py) unknown optim name, adam will be used')
            return torch.optim.Adam(self.parameters(), lr=self.cfg_optimizer.BASE_LR, betas=(0.9, 0.999), eps=1e-08)
    
    def training_epoch_end(self, outputs) -> None:
        if self.monitor_grad:
            print('X_fit grads = 0?',(self.grads['compressed_X_fit'] == 0.0).all())
            print('norm y_fit grads = 0?',(self.grads['norm compressed_y_fit'] == 0.0).all())
            print('ceil y_fit grads = 0?',(self.grads['ceil compressed_y_fit'] == 0.0).all())
        # print('X_fit grad:\n{}\n'.format(self.grads['compressed_X_fit']))
        # print('norm y_fit grad:\n{}\n'.format(self.grads['norm compressed_y_fit']))
        # print('ceil y_fit grad:\n{}\n'.format(self.grads['ceil compressed_y_fit']))