class CompressionNetForKRR(pl.LightningModule):
    def __init__(self, conv_module, X_fit, coef_fit, n_compressed, params, cfg_optimizer):
        super().__init__()
        
        self.conv = conv_module
        self.fx_linear = nn.Linear(n_compressed, 1, bias=False) # weights(n_compressed, 1), bias(1,)
        # initial fx_linear, the weights is compressed_coef, the bia is intercept, but krr don't have intercept
        # 用coef_fit初始化
        self.fx_linear.weight = nn.Parameter(torch.Tensor((coef_fit.reshape(-1,1))[:n_compressed].reshape(1,-1)))
        self.register_buffer('X_fit', X_fit)
        self.cfg_optimizer = cfg_optimizer
        self.params = params
        # init criterion
        self.criterion = nn.L1Loss()

        self.grads = {}
        self.monitor_grad = False

    def forward(self):
        # return compressed_X_fit, compressed_coef, intercept
        compressed_X_fit = (self.conv(self.X_fit)).cpu().detach().numpy()
        compressed_coef = self.fx_linear.weight.data.cpu().detach().numpy()
        results = {'compressed_X_fit':compressed_X_fit, 'compressed_coef':compressed_coef}
        return results

    def training_step(self, batch, batch_idx): 
        gc.collect()
        X, y = batch
        if self.monitor_grad:
            print('X',X.requires_grad)
        compressed_X_fit = self.conv(self.X_fit)
        if self.monitor_grad:
            print('x fit:',compressed_X_fit.requires_grad)
        km = self.cal_km(compressed_X_fit, X) # shape: (n_compressed, batch_size)
        if self.monitor_grad:
            print('km:',km.requires_grad)
        # 由于linear.weight是(1,n_compressed)，计算xA^T + b，因此km要转置
        # (batch_size, n_gen) @ (n_gen, 1) = (batch_size , 1) 即fx 
        fx = self.fx_linear(km.T)
        if self.monitor_grad:
            print('fx:',fx.requires_grad)
            self.monitor_grad=False
        fx = fx.view(-1,)
        # loss = F.binary_cross_entropy_with_logits(fx, y)
        loss = self.criterion(fx, y)
        self.log('train_loss', loss)
        del km
        gc.collect()

        # register hook
        if self.monitor_grad:
            compressed_X_fit.register_hook(save_grad('compressed_X_fit',self.grads))
            fx.register_hook(save_grad('fx',self.grads))
            loss.register_hook(save_grad('loss',self.grads))

        return loss

    def validation_step(self, batch, batch_idx):
        gc.collect()
        X, y = batch
        compressed_X_fit = self.conv(self.X_fit)
        km = self.cal_km(compressed_X_fit, X) # shape: (n_compressed, batch_size)
        # 由于linear.weight是(1,n_compressed)，计算xA^T + b，因此km要转置
        # (batch_size, n_gen) @ (n_gen, 1) = (batch_size , 1) 即fx 
        # print((km.T).size())
        # print(self.fx_linear.weight.size())
        fx = self.fx_linear(km.T)
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

    def cal_km(self, X_fit, X):
        if self.params['kernel'] == 'linear':
            return torch.mm(X_fit, X.t())
        elif self.params['kernel'] == 'rbf':
            # X_a = X.view(-1, 1, X.size(1))
            # km = torch.exp(-self.params['gamma'] * torch.sum(torch.pow(X_fit - X_a, 2), axis=2))
            km = torch.exp(-self.params['gamma'] * torch.cdist(X_fit, X, p=2)**2)
            return km #.t()
        elif self.params['kernel'] == 'poly':
            return torch.pow(self.params['gamma'] * torch.mm(X_fit, X.t()) + self.params['coef0'], self.params['degree'])
        elif self.params['kernel'] == 'sigmoid':
            return torch.tanh(self.params['gamma'] * torch.mm(X_fit, X.t()) + self.params['coef0'])
        else:
            warnings.warn('(compression_net.py) unknown kernel, can not do cal_km function')
            return