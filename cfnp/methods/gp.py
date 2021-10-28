class CompressionNetForGP(pl.LightningModule):
    def __init__(self, conv_module, X_fit, y_fit, n_compressed, params, cfg_optimizer):
        super().__init__()
        
        self.conv = conv_module
        # 用y_fit初始化
        self.compressed_y_fit = nn.Parameter(torch.Tensor((y_fit.reshape(-1,1))[:n_compressed]))
        self.register_buffer('X_fit', X_fit)
        self.cfg_optimizer = cfg_optimizer
        self.params = params
        # init criterion
        self.criterion = SumOfKlLoss()

        self.grads = {}
        self.monitor_grad = False

    def forward(self):
        # return compressed_X_fit, compressed_coef, intercept
        compressed_X_fit = (self.conv(self.X_fit)).cpu().detach().numpy()
        compressed_y_fit = self.compressed_y_fit.data.cpu().detach().numpy()
        results = {'compressed_X_fit':compressed_X_fit, 'compressed_y_fit':compressed_y_fit}
        return results

    def training_step(self, batch, batch_idx): 
        X, y = batch
        mean, std = y.split(1, dim=1)
        mean = mean.view(-1,1)
        std = std.view(-1,1)

        compressed_X_fit = self.conv(self.X_fit)
        K = self.cal_km(compressed_X_fit, compressed_X_fit, constant_value=self.params['constant_value'], length_scale=self.params['length_scale']) 
        diag_eps = self.params['alpha'] * torch.eye(K.size(0)).type_as(K)
        K = K + diag_eps
        L = torch.linalg.cholesky(K)
        alpha = torch.cholesky_solve(self.compressed_y_fit, L).view(-1,1)
        K_trans = self.cal_km(compressed_X_fit, X, constant_value=self.params['constant_value'], length_scale=self.params['length_scale'])

        mean_pred = torch.mm(K_trans, alpha)
        mean_pred = self.params['y_train_std'] * mean_pred + self.params['y_train_mean']
        mean_pred = mean_pred.view(-1,1)

        V = torch.cholesky_solve(K_trans.T, L)
        var_pred = torch.ones(X.size(0)).type_as(V)
        var_pred -= torch.einsum("ij,ji->i", K_trans, V)
        var_pred = torch.clamp(var_pred, min=0.0)
        var_pred = var_pred * self.params['y_train_std']
        std_pred = torch.sqrt(var_pred)
        std_pred = std_pred.view(-1,1)

        # y_pred = torch.cat([mean_pred, std_pred], dim=-1)

        loss = self.criterion(mean, std, mean_pred, std_pred)
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        mean, std = y.split(1, dim=1)
        mean = mean.view(-1,1)
        std = std.view(-1,1)

        compressed_X_fit = self.conv(self.X_fit)
        K = self.cal_km(compressed_X_fit, compressed_X_fit, constant_value=self.params['constant_value'], length_scale=self.params['length_scale']) 
        diag_eps = self.params['alpha'] * torch.eye(K.size(0)).type_as(K)
        K = K + diag_eps
        L = torch.linalg.cholesky(K)
        alpha = torch.cholesky_solve(self.compressed_y_fit, L).view(-1,1)
        K_trans = self.cal_km(compressed_X_fit, X, constant_value=self.params['constant_value'], length_scale=self.params['length_scale'])

        mean_pred = torch.mm(K_trans, alpha)
        mean_pred = self.params['y_train_std'] * mean_pred + self.params['y_train_mean']
        mean_pred = mean_pred.view(-1,1)

        V = torch.cholesky_solve(K_trans.T, L)
        var_pred = torch.ones(X.size(0)).type_as(V)
        var_pred -= torch.einsum("ij,ji->i", K_trans, V)
        var_pred = torch.clamp(var_pred, min=0.0)
        var_pred = var_pred * self.params['y_train_std']
        std_pred = torch.sqrt(var_pred)
        std_pred = std_pred.view(-1,1)

        # y_pred = torch.cat([mean_pred, std_pred], dim=-1)

        loss = self.criterion(mean, std, mean_pred, std_pred)
        self.log('train_loss', loss)

        return loss

    def configure_optimizers(self):
        if self.cfg_optimizer.NAME.lower() == 'adam':
            return torch.optim.Adam(self.parameters(), lr=self.cfg_optimizer.BASE_LR, betas=(0.9, 0.999), eps=1e-08)
        else:
            warnings.warn('(compression_net.py) unknown optim name, adam will be used')
            return torch.optim.Adam(self.parameters(), lr=self.cfg_optimizer.BASE_LR, betas=(0.9, 0.999), eps=1e-08)
    
    def cal_km(self, X, Y, length_scale, constant_value):
        X_scale = X/length_scale
        Y_scale = Y/length_scale
        dists = torch.cdist(X_scale, Y_scale, p=2) **2
        K = torch.exp(-.5 * dists)
        K = constant_value * K
        return K.T