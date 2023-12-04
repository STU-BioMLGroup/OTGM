import torch
import torch.nn as nn
import evaluation

class Autoencoder(nn.Module):
    """AutoEncoder module that projects features to latent space."""
    def __init__(self,
                 encoder_dim,
                 activation='relu',
                 batchnorm=True):
        """Constructor.

        Args:
          encoder_dim: Should be a list of ints, hidden sizes of
            encoder network, the last element is the size of the latent representation.
          activation: Including "sigmoid", "tanh", "relu", "leakyrelu". We recommend to
            simply choose relu.
          batchnorm: if provided should be a bool type. It provided whether to use the
            batchnorm in autoencoders.
        """
        super(Autoencoder, self).__init__()

        self._dim = len(encoder_dim) - 1
        self._activation = activation
        self._batchnorm = batchnorm

        encoder_layers = []
        for i in range(self._dim):
            encoder_layers.append(
                nn.Linear(encoder_dim[i], encoder_dim[i + 1]))
            if i < self._dim - 1:
                if self._batchnorm:
                    encoder_layers.append(nn.BatchNorm1d(encoder_dim[i + 1]))
                if self._activation == 'sigmoid':
                    encoder_layers.append(nn.Sigmoid())
                elif self._activation == 'leakyrelu':
                    encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == 'tanh':
                    encoder_layers.append(nn.Tanh())
                elif self._activation == 'relu':
                    encoder_layers.append(nn.ReLU())
                else:
                    raise ValueError('Unknown activation type %s' % self._activation)
        encoder_layers.append(nn.Softmax(dim=1))
        self._encoder = nn.Sequential(*encoder_layers)

        decoder_dim = [i for i in reversed(encoder_dim)]
        decoder_layers = []
        for i in range(self._dim):
            decoder_layers.append(
                nn.Linear(decoder_dim[i], decoder_dim[i + 1]))
            if self._batchnorm:
                decoder_layers.append(nn.BatchNorm1d(decoder_dim[i + 1]))
            if self._activation == 'sigmoid':
                decoder_layers.append(nn.Sigmoid())
            elif self._activation == 'leakyrelu':
                decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self._activation == 'tanh':
                decoder_layers.append(nn.Tanh())
            elif self._activation == 'relu':
                decoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self._activation)
        self._decoder = nn.Sequential(*decoder_layers)

    def encoder(self, x):
        """Encode sample features.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [n_nodes, latent_dim] float tensor, representation Z.
        """
        latent = self._encoder(x)
        return latent

    def decoder(self, latent):
        """Decode sample features.

            Args:
              latent: [num, latent_dim] float tensor, representation Z.

            Returns:
              x_hat: [n_nodes, feat_dim] float tensor, reconstruction x.
        """
        x_hat = self._decoder(latent)
        return x_hat

    def forward(self, x):
        """Pass through autoencoder.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [num, latent_dim] float tensor, representation Z.
              x_hat:  [num, feat_dim] float tensor, reconstruction x.
        """
        latent = self.encoder(x)
        x_hat = self.decoder(latent)
        return x_hat, latent


class Generator(nn.Module):
    """Dual prediction module that projects features from corresponding latent space."""

    def __init__(self, feature_dim, activation='relu', batchnorm=True):
        """Constructor.

        Args:
          prediction_dim: Should be a list of ints, hidden sizes of
            prediction network, the last element is the size of the latent representation of autoencoder.
          activation: Including "sigmoid", "tanh", "relu", "leakyrelu". We recommend to
            simply choose relu.
          batchnorm: if provided should be a bool type. It provided whether to use the
            batchnorm in autoencoders.
        """
        super(Generator, self).__init__()

        self._depth = len(feature_dim) - 1
        self._activation = activation
        self._prediction_dim = feature_dim

        encoder_layers = []
        for i in range(self._depth):
            encoder_layers.append(
                nn.Linear(self._prediction_dim[i], self._prediction_dim[i + 1]))
            if batchnorm:
                encoder_layers.append(nn.BatchNorm1d(self._prediction_dim[i + 1]))
            if self._activation == 'sigmoid':
                encoder_layers.append(nn.Sigmoid())
            elif self._activation == 'leakyrelu':
                encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self._activation == 'tanh':
                encoder_layers.append(nn.Tanh())
            elif self._activation == 'relu':
                encoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self._activation)
        self._encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        for i in range(self._depth, 0, -1):
            decoder_layers.append(
                nn.Linear(self._prediction_dim[i], self._prediction_dim[i - 1]))
            if i > 1:
                if batchnorm:
                    decoder_layers.append(nn.BatchNorm1d(self._prediction_dim[i - 1]))
                if self._activation == 'sigmoid':
                    decoder_layers.append(nn.Sigmoid())
                elif self._activation == 'leakyrelu':
                    decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == 'tanh':
                    decoder_layers.append(nn.Tanh())
                elif self._activation == 'relu':
                    decoder_layers.append(nn.ReLU())
                else:
                    raise ValueError('Unknown activation type %s' % self._activation)
        decoder_layers.append(nn.Softmax(dim=1))
        self._decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        """Data recovery by prediction.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [num, latent_dim] float tensor.
              output:  [num, feat_dim] float tensor, recovered data.
        """
        latent = self._encoder(x)
        output = self._decoder(latent)
        return output, latent

class Network(nn.Module):
    def __init__(self,
                 config):
        super(Network, self).__init__()
        if config['Autoencoder']['arch1'][-1] != config['Autoencoder']['arch2'][-1]:
            raise ValueError('Inconsistent latent dim!')

        self._latent_dim = config['Autoencoder']['arch1'][-1]
        self._num_sample = config['training']['num_sample']
        self._num_classes = config['training']['num_classes']
        self._dims_view1 = [self._latent_dim] + config['Prediction']['arch1']
        self._dims_view2 = [self._latent_dim] + config['Prediction']['arch2']

        # View-specific autoencoders
        self.autoencoder1 = Autoencoder(config['Autoencoder']['arch1'], config['Autoencoder']['activations1'],
                                        config['Autoencoder']['batchnorm'])
        self.autoencoder2 = Autoencoder(config['Autoencoder']['arch2'], config['Autoencoder']['activations2'],
                                        config['Autoencoder']['batchnorm'])
        self.generator1 = Generator(self._dims_view1)
        self.generator2 = Generator(self._dims_view2)

    def forward(self, x1_train, x2_train):
        z1 = self.autoencoder1.encoder(x1_train)
        z2 = self.autoencoder2.encoder(x2_train)
        z1_hat, _ = self.generator1(z1.detach())
        z2_hat, _ = self.generator2(z2.detach())
        x1_recon = self.autoencoder1.decoder(z1)
        x2_recon = self.autoencoder2.decoder(z2)
        return x1_recon, x2_recon, z1, z2, z1_hat, z2_hat

    def evaluation(self, logger, x1_train, x2_train, Y_list):
        with torch.no_grad():
            self.autoencoder1.eval(), self.autoencoder2.eval()
            self.generator1.eval(), self.generator2.eval()
            z1 = self.autoencoder1.encoder(x1_train)
            z2 = self.autoencoder2.encoder(x2_train)
            z1_hat, _ = self.generator1(z1)
            z2_hat, _ = self.generator2(z2)
            latent_fusion = torch.cat([z2_hat, z1_hat], dim=1).cpu().numpy()
            scores = evaluation.clustering([latent_fusion], Y_list[0])
            logger.info("\033[2;29m" + '     ===>pretrain ' + str(scores) + "\033[0m")
            self.autoencoder1.train(), self.autoencoder2.train()
            self.generator1.train(), self.generator2.train()
        return scores

class GOT(nn.Module):
    def __init__(self, nodes, tau, it):
        super(GOT, self).__init__()
        self._nodes = nodes
        self._tau = tau
        self._it = it
        self.mean = nn.Parameter(torch.rand((self._nodes, self._nodes), dtype=torch.float32), requires_grad=True)
        self.std = nn.Parameter(10 * torch.ones((self._nodes, self._nodes), dtype=torch.float32), requires_grad=True)

    def init_param(self, similarity):
        self.mean.data = similarity

    def doubly_stochastic(self, P):
        """Uses logsumexp for numerical stability."""
        A = P / self._tau
        for i in range(self._it):
            A = A - A.logsumexp(dim=1, keepdim=True)
            A = A - A.logsumexp(dim=0, keepdim=True)
        return torch.exp(A)

    def forward(self, eps):
        P_noisy = self.mean + self.std * eps
        DS = self.doubly_stochastic(P_noisy)
        return DS

    def loss_got(self, g1, g2, DS, params):
        [C1_tilde, C2_tilde] = params
        loss_c = torch.trace(g1) + torch.trace(DS @ g2 @ torch.transpose(DS, 0, 1))
        # svd version
        u, sigma, v = torch.svd(C2_tilde @ torch.transpose(DS, 0, 1) @ C1_tilde)
        loss = loss_c - 2 * torch.sum(sigma)
        return loss

class OTGM(nn.Module):
    def __init__(self, config):
        super(OTGM, self).__init__()
        if config['Autoencoder']['arch1'][-1] != config['Autoencoder']['arch2'][-1]:
            raise ValueError('Inconsistent latent dim!')
        self._num_sample = config['training']['num_sample']
        self._num_aligned = int(self._num_sample * config['training']['aligned_ratio'])
        self._num_mis_aligned = self._num_sample - self._num_aligned

        self.network = Network(config)
        self.got = GOT(self._num_mis_aligned, config['training']['got']['tau'], config['training']['got']['it'])

    def forward(self, x1_train, x2_train):
        return self.network(x1_train, x2_train)



