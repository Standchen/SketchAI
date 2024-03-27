import torch
import torch.nn as nn
import torch.nn.functional as F

# torchvision
import torchvision.transforms as transforms

from utils import IMG_MEAN, IMG_STD


class MyConv(nn.Module):
    """
    Convolutional layer that integrates optional normalization and activation layers.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, padding_mode="zeros", bias=False,
                 transpose=False, normalization="batch", relu="relu"):
        super().__init__()
        # Convolutional layer
        if transpose:
            self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                           kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode, bias=bias)
        else:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode, bias=bias)

        # Normalization
        if normalization == "batch":
            self.norm = nn.BatchNorm2d(num_features=out_channels)
        elif normalization == "instance":
            if not bias:
                print("[*] It is recommended to use bias if the layer is instance-normalized.")
            self.norm = nn.InstanceNorm2d(num_features=out_channels)
        elif normalization is None:
            self.norm = nn.Identity()
        else:
            raise ValueError

        # Activation
        if relu == "relu":
            self.relu = nn.ReLU(inplace=True)
        elif isinstance(relu, (float, int)):
            self.relu = nn.LeakyReLU(relu, inplace=True)
        elif relu is None:
            self.relu = nn.Identity()
        else:
            raise ValueError

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class ResidualBlock(nn.Module):
    """
    Simple redisual block.
    """

    def __init__(self, in_dim, mid_dim, out_dim, use_1x1conv=False, use_dropout=False):
        super().__init__()
        self.block = []

        # Convolutional layers
        self.block += [MyConv(in_dim, mid_dim, padding=1, padding_mode="reflect")]
        self.block += [nn.Dropout(0.5)] if use_dropout else []
        self.block += [MyConv(mid_dim, out_dim, padding=1, padding_mode="reflect",
                              relu=None)]

        self.block = nn.Sequential(*self.block)

        # If in_dim != out_dim, it should assert to use 1x1 convolution.
        assert use_1x1conv or in_dim == out_dim
        if use_1x1conv:
            self.skip = MyConv(in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=True,
                               normalization=None, relu=None)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        return F.relu(self.skip(x) + self.block(x))


class GeneratorModel(nn.Module):
    """
    Tweaked generator from `Learning to Simplify: Fully Convolutional Networks for Rough Sketch Cleanup`.
    """
    def __init__(self, input_nc: int, output_nc: int, ngf: int, num_residual_blk: int):
        super().__init__()
        self.generator = []
        self.generator += [MyConv(input_nc, 1*ngf, kernel_size=5, padding=2, padding_mode="reflect"),
                           ResidualBlock(in_dim=1*ngf, mid_dim=1*ngf, out_dim=1*ngf),

                           MyConv(1*ngf, 2*ngf, kernel_size=3, stride=2, padding=1, padding_mode="reflect"),
                           ResidualBlock(in_dim=2*ngf, mid_dim=2*ngf, out_dim=2*ngf),

                           MyConv(2*ngf, 4*ngf, kernel_size=3, stride=2, padding=1, padding_mode="reflect")]

        self.generator += [ResidualBlock(in_dim=4*ngf, mid_dim=4*ngf, out_dim=4*ngf) for _ in range(num_residual_blk)]

        self.generator += [MyConv(4*ngf, 4*ngf, kernel_size=4, stride=2, padding=1, transpose=True),
                           ResidualBlock(in_dim=4*ngf, mid_dim=4*ngf, out_dim=2*ngf, use_1x1conv=True),

                           MyConv(2*ngf, 2*ngf, kernel_size=4, stride=2, padding=1, transpose=True),
                           ResidualBlock(in_dim=2*ngf, mid_dim=2*ngf, out_dim=1*ngf, use_1x1conv=True),

                           ResidualBlock(in_dim=1*ngf, mid_dim=1*ngf, out_dim=ngf//2, use_1x1conv=True),

                           MyConv(ngf//2, output_nc, kernel_size=5, padding=2, padding_mode="reflect",
                                  normalization=None, relu=None),

                           nn.Tanh()]

        self.generator = nn.Sequential(*self.generator)
        
        self.normalizer = transforms.Normalize(mean=(IMG_MEAN,), std=(IMG_STD,))
    
    def forward(self, x):
        # return self.generator(x)
        return self.normalizer(self.generator(x))


class DiscriminatorModel(nn.Module):
    """
    PatchGAN discriminator.
    """

    def __init__(self, input_nc: int, ndf: int, nlayers: int):
        super().__init__()
        self.model = []
        self.model += [MyConv(2*input_nc, 1*ndf, kernel_size=4, stride=2, padding=1,
                              normalization=None, relu=0.2)]

        k = 1
        for i in range(1, nlayers):
            prev_k = k
            k = min(2**i, 8)
            self.model += [MyConv(prev_k*ndf, k*ndf, kernel_size=4, stride=2, padding=1, relu=0.2)]
        
        self.model += [nn.Dropout(0.5)]

        prev_k = k
        k = min(2**nlayers, 8)
        self.model += [MyConv(prev_k*ndf, k*ndf, kernel_size=4, stride=1, padding=1, relu=0.2)]

        self.model += [nn.Dropout(0.5)]

        self.model += [MyConv(k*ndf, 1, kernel_size=4, stride=1, padding=1,
                              normalization=None, relu=None)]

        self.model = nn.Sequential(*self.model)

    def forward(self, a, b):
        return self.model(torch.concat([a, b], dim=1))

