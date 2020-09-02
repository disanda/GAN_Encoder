import torch
import torch.nn as nn

class Identity(nn.Module):
    def __init__(self, *args, **keyword_args):
        super().__init__()
    def forward(self, x):
        return x

def _get_norm_layer_2d(norm):
    if norm == 'none':
        return Identity
    elif norm == 'batch_norm':
        return nn.BatchNorm2d
    elif norm == 'instance_norm':
        return functools.partial(nn.InstanceNorm2d, affine=True)
    elif norm == 'layer_norm':
        return lambda num_features: nn.GroupNorm(1, num_features)
    else:
        raise NotImplementedError

# z:[128,-1,1,1]  , y/x = [-1,1,64,64]

# class D(nn.Module):
#     def __init__(self, nc, ndf):
#         super().__init__()
#         self.main = nn.Sequential(
#             nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),             # input is (nc) x 64 x 64
#             nn.LeakyReLU(0.2, inplace=True),

#             nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),                   # state size. (ndf) x 32 x 32
#             nn.BatchNorm2d(ndf * 2),
#             nn.LeakyReLU(0.2, inplace=True),

#             nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),                           # state size. (ndf*2) x 16 x 16
#             nn.BatchNorm2d(ndf * 4),
#             nn.LeakyReLU(0.2, inplace=True),

#             nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),                                       # state size. (ndf*4) x 8 x 8
#             nn.BatchNorm2d(ndf * 8),
#             nn.LeakyReLU(0.2, inplace=True),

#             nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),             # state size. (ndf*8) x 4 x 4
#             nn.Sigmoid()
#         )
#     def forward(self, input):
#         return self.main(input)

#与上面一致
class ConvDiscriminator(nn.Module):
    def __init__(self,
                 input_channels=1,
                 dim=128,
                 n_downsamplings=4,
                 norm='batch_norm'):
        super().__init__()

        Norm = _get_norm_layer_2d(norm)

        def conv_norm_lrelu(in_dim, out_dim, kernel_size=4, stride=2, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding, bias=False or Norm == Identity),
                Norm(out_dim),
                nn.LeakyReLU(0.2)
            )
        layers = []
        # 1: downsamplings, ... -> 16x16 -> 8x8 -> 4x4
        d = dim
        layers.append(nn.Conv2d(input_channels, d, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2))
        for i in range(n_downsamplings - 1):
            d_last = d
            d = min(dim * 2 ** (i + 1), dim * 8)
            layers.append(conv_norm_lrelu(d_last, d, kernel_size=4, stride=2, padding=1))
        # 2: logit
        layers.append(nn.Conv2d(d, 1, kernel_size=4, stride=1, padding=0))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        y = self.net(x)
        return y #[1,1,1,1]


class encoder_v3(nn.Module):
    def __init__(self,
                 input_channels=1,
                 dim=128,
                 n_downsamplings=4,
                 norm='batch_norm'):
        super().__init__()

        Norm = _get_norm_layer_2d(norm)

        def conv_norm_lrelu(in_dim, out_dim, kernel_size=4, stride=2, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding, bias=False or Norm == Identity),
                Norm(out_dim),
                nn.LeakyReLU(0.2)
            )
        layers = []
        # 1: downsamplings, ... -> 16x16 -> 8x8 -> 4x4
        d = dim
        layers.append(nn.Conv2d(input_channels, d, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2))
        for i in range(n_downsamplings - 1):
            d_last = d
            d = min(dim * 2 ** (i + 1), dim * 8)
            layers.append(conv_norm_lrelu(d_last, d, kernel_size=4, stride=2, padding=1))
        # 2: logit
        layers.append(nn.Conv2d(d, 128, kernel_size=4, stride=1, padding=0)) #只改动这一层
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        y = self.net(x)
        y = torch.transpose(y, 1, 0) #[-1,128,1,1]->[128,-1,1,1] 对齐G
        return y 

# #encoder_v1: linear+lrelu
# class VAE(nn.Module):
#     def __init__(self):
#         super(VAE, self).__init__()er

#         self.fc1 = nn.Linear(784, 400)
#         self.fc21 = nn.Linear(400, 20)
#         self.fc22 = nn.Linear(400, 20)
        
#         self.fc3 = nn.Linear(20, 400)
#         self.fc4 = nn.Linear(400, 784)

#     def encode(self, x):
#         h1 = F.relu(self.fc1(x))
#         return self.fc21(h1), self.fc22(h1)

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5*logvar)
#         eps = torch.randn_like(std)#0-1随机数
#         return mu + eps*std

#     def decode(self, z):
#         h3 = F.relu(self.fc3(z))
#         return torch.sigmoid(self.fc4(h3))

#     def forward(self, x):
#         mu, logvar = self.encode(x.view(-1, 784))
#         z = self.reparameterize(mu, logvar)
#         return self.decode(z), mu, logvar


# #encoder_v2: conv+linear+lrelu
#     def encoder(self, x, is_training=True, reuse=False):
#         # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
#         # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC62*4
#         with tf.variable_scope("encoder", reuse=reuse):

#             net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='en_conv1'))
#             net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='en_conv2'), is_training=is_training, scope='en_bn2'))
#             net = tf.reshape(net, [self.batch_size, -1])
#             net = lrelu(bn(linear(net, 1024, scope='en_fc3'), is_training=is_training, scope='en_bn3'))
#             gaussian_params = linear(net, 2 * self.z_dim, scope='en_fc4')

#             # The mean parameter is unconstrained
#             mean = gaussian_params[:, :self.z_dim]
#             # The standard deviation must be positive. Parametrize with a softplus and
#             # add a small epsilon for numerical stability
#             stddev = 1e-6 + tf.nn.softplus(gaussian_params[:, self.z_dim:])

#         return mean, stddev


