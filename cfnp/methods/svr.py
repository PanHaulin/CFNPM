import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import warnings
from methods.base import RegressionBaseModel

class CompressionNetForSVR(RegressionBaseModel):
    def __init__(self, conv_module, X_fit, coef_fit, intercept, n_compressed, params, cfg_optimizer, model_type, ratio, data):
        super().__init__()
        
        self.conv = conv_module
        self.fx_linear = nn.Linear(n_compressed, 1, bias=True) # weights(n_compressed, 1), bias(1,)
        # self.sigmoid = nn.Sigmoid()
        # initial fx_linear, the weights is compressed_coef, the bia is intercept
        # 用coef_fit和intercept初始化
        self.fx_linear.weight = nn.Parameter(torch.Tensor((coef_fit.reshape(-1,1))[:n_compressed].reshape(1,-1)))
        self.fx_linear.bias = nn.Parameter(torch.Tensor([intercept]))
        # self.fx_linear.weight = nn.Parameter(torch.rand(1,n_compressed))
        # self.fx_linear.bias = nn.Parameter(torch.Tensor([1]))
        # self.compressed_X_fit = nn.Parameter(torch.rand(n_compressed,123))
        #TODO:修改成有约束形式
        # 用xavier正态分布初始化
        # nn.init.xavier_uniform(self.fx_linear.weight)
        # nn.init.xavier_uniform(self.fx_linear.bias)
        self.register_buffer('X_fit', X_fit)

        self.cfg_optimizer = cfg_optimizer
        self.params = params
        # init criterion
        if model_type == 'svc':
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.L1Loss()

        self.grads = {}
        self.monitor_grad = False
        self.check = False

        # 统计信息
        self.n_epochs=0
        self.n_fit = X_fit.size(2)
        self.n_compressed = n_compressed
        self.n_features = X_fit.size(1) + 1 # n_features + coef
        self.ratio = ratio

        # 预测用信息
        self.X_train = data['X_train']
        self.X_test = data['X_test']
        self.y_train = data['y_train']
        self.y_test = data['y_test']
        self.model_type = model_type

    def constrained_linear_fx(self, km):
        if self.check:
            print('km:{}'.format(km.size())) # (batch_size, n_compressed)
        alpha = F.softmax(torch.abs(self.fx_linear.weight), dim=0) #行处理
        if self.check:
            print("alpha:{}".format(alpha.size())) # (1, n_compressed)
        coef = alpha * torch.sign(self.fx_linear.weight)
        if self.check:
            print('coef:{}'.format(coef.size())) # (1, n_compressed)
        fx = torch.mm(km, coef.T) + self.fx_linear.bias
        if self.check:
            print('fx:{}'.format(fx.size())) # (batch_size, 1)
            self.check=False
        
        return fx

    def get_coef(self):
        alpha = F.softmax(torch.abs(self.fx_linear.weight), dim=0)
        coef = alpha * torch.sign(self.fx_linear.weight)
        return coef

    def get_compression_results(self):
        # return compressed_X_fit, compressed_coef, intercept
        compressed_X_fit = (self.conv(self.X_fit)).cpu().detach().numpy()
        # compressed_X_fit = self.compressed_X_fit.cpu().detach().numpy()
        # compressed_coef = self.fx_linear.weight.data.cpu().detach().numpy()
        compressed_coef = self.get_coef().data.cpu().detach().numpy()
        compressed_intercept = self.fx_linear.bias.data.cpu().detach().numpy()
        results = {'compressed_X_fit':compressed_X_fit, 'compressed_coef':compressed_coef, 'compressed_intercept':compressed_intercept[0]}
        return results

    def training_step(self, batch, batch_idx): 
        gc.collect()
        X, y = batch
        # if self.monitor_grad:
        #     print('X',X.requires_grad)
        compressed_X_fit = self.conv(self.X_fit)
        # compressed_X_fit = self.compressed_X_fit
        # if self.monitor_grad:
        #     print('x fit:',compressed_X_fit.requires_grad)
        km = self.cal_km(compressed_X_fit, X) # shape: (n_compressed, batch_size)
        # if self.monitor_grad:
        #     print('km:',km.requires_grad)
        # 由于linear.weight是(1,n_compressed)，计算xA^T + b，因此km要转置
        # (batch_size, n_compressed) @ (n_compressed, 1) = (batch_size , 1) 即fx 
        # fx = self.fx_linear(km.T)
        fx = self.constrained_linear_fx(km.T)
        # if self.monitor_grad:
        #     print('fx:',fx.requires_grad)
        #     self.monitor_grad=False
        fx = fx.view(-1,)
        # loss = F.binary_cross_entropy_with_logits(fx, y)
        loss = self.criterion(fx, y)
        self.log('train_loss', loss)
        del km
        gc.collect()

        # register hook
        # if self.monitor_grad:
        #     compressed_X_fit.register_hook(save_grad('compressed_X_fit',self.grads))
        #     fx.register_hook(save_grad('fx',self.grads))
        #     loss.register_hook(save_grad('loss',self.grads))

        return loss

    def validation_step(self, batch, batch_idx):
        gc.collect()
        X, y = batch
        compressed_X_fit = self.conv(self.X_fit)
        # compressed_X_fit = self.compressed_X_fit
        km = self.cal_km(compressed_X_fit, X) # shape: (n_compressed, batch_size)
        # 由于linear.weight是(1,n_compressed)，计算xA^T + b，因此km要转置
        # (batch_size, n_gen) @ (n_gen, 1) = (batch_size , 1) 即fx 
        # fx = self.fx_linear(km.T)
        fx = self.constrained_linear_fx(km.T)
        fx = fx.view(-1,)

        # loss = F.binary_cross_entropy_with_logits(fx, y)
        loss = self.criterion(fx, y)
        self.log('valid_loss', loss)
        del km
        gc.collect()

    def configure_optimizers(self):
        if self.cfg_optimizer.NAME.lower() == 'adam':
            return torch.optim.Adam(self.parameters(), lr=self.cfg_optimizer.BASE_LR, betas=(0.9, 0.999), eps=1e-08)
        else:
            warnings.warn('(compression_net.py) unknown optim name, adam will be used')
            return torch.optim.Adam(self.parameters(), lr=self.cfg_optimizer.BASE_LR, betas=(0.9, 0.999), eps=1e-08)

    # def count(self, X, y):
    #     compressed_X_fit = self.conv(self.X_fit)
    #     # compressed_X_fit = self.compressed_X_fit
    #     km = self.cal_km(compressed_X_fit, X) # shape: (n_compressed, batch_size)
    #     # 由于linear.weight是(1,n_compressed)，计算xA^T + b，因此km要转置
    #     # (batch_size, n_gen) @ (n_gen, 1) = (batch_size , 1) 即fx 
    #     # fx = self.fx_linear(km.T)
    #     fx = self.constrained_linear_fx(km.T)
    #     fx = fx.view(-1,)
    #     # loss = F.binary_cross_entropy_with_logits(fx, y)
    #     loss = self.criterion(fx, y)

    def cal_km(self, X_fit, X):
        if self.params['kernel'] == 'linear':
            return torch.mm(X_fit, X.t())
        elif self.params['kernel'] == 'rbf':
            # X_a = X.view(-1, 1, X.size(1))
            # km = torch.exp(-self.params['gamma'] * torch.sum(torch.pow(X_fit - X_a, 2), axis=2))
            km = torch.exp(-self.params['gamma'] * torch.cdist(X_fit, X, p=2)**2)
            return km #.t()
            # km = torch.Tensor(X_fit.size(0), X.size(0))
            # km.type_as(X_fit)
            # for i in range(X_fit.size(0)):
            #     for j in range(X.size(0)):
            #         km[i][j] = torch.exp(-self.params['gamma']*torch.sum(torch.pow(X_fit[i]-X[j], 2)))
            # return km.cuda()
        elif self.params['kernel'] == 'poly':
            return torch.pow(self.params['gamma'] * torch.mm(X_fit, X.t()) + self.params['coef0'], self.params['degree'])
        elif self.params['kernel'] == 'sigmoid':
            return torch.tanh(self.params['gamma'] * torch.mm(X_fit, X.t()) + self.params['coef0'])
        else:
            warnings.warn('(compression_net.py) unknown kernel, can not do cal_km function')
            return

    def on_train_epoch_start(self):   
        start = time.time()
        compression_results = self.get_compression_results()
        # log_compressed_info
        eval_results = eval_compression_ins(self.X_train, self.X_test, self.y_train, self.y_test, self.model_type, compression_results, self.params)
        if self.model_type == 'svc':
            self.log('train_acc', eval_results['trainset_report_dict']['acc'], on_epoch=True)
            self.log('train_auc', eval_results['trainset_report_dict']['auc'], on_epoch=True)
            self.log('test_acc', eval_results['testset_report_dict']['acc'], on_epoch=True)
            self.log('test_auc', eval_results['testset_report_dict']['auc'], on_epoch=True)
        else:
            self.log('train_mae', eval_results['trainset_report_dict']['mae'], on_epoch=True)
            self.log('train_mse', eval_results['trainset_report_dict']['mse'], on_epoch=True)
            self.log('test_mae', eval_results['testset_report_dict']['mae'], on_epoch=True)
            self.log('test_mse', eval_results['testset_report_dict']['mse'], on_epoch=True)
        end = time.time()
        self.log('eval_time', end-start, on_epoch=True)
        
    def on_train_end(self):
        # if self.n_epochs == self.max_epochs:
        #     # print('max_epochs')
        # log eval
        self.on_train_start()

        # log ratio
        self.logger.log_metric('ac_ratio', 1- (self.n_compressed / self.n_fit))

        # log n_ins
        self.logger.log_metrics({'bf_n_ins':self.n_fit, 'af_n_ins':self.n_compressed})

        # log bytes
        before_size = self.n_features * self.n_fit * 8 /1024
        after_size = self.n_features * self.n_compressed * 8 / 1024
        self.logger.log_metrics({'bf_size(MB)':before_size, 'af_size(MB)':after_size})