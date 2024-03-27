from collections import OrderedDict

import torch
import torch.nn as nn

from utils import IMG_MEAN, IMG_STD


PRETRAINED_VGG19_PATH = "./data/vgg_conv.pth"

VGG_MEAN = (0.485, 0.456, 0.406)
VGG_STD  = (0.229, 0.224, 0.225)


class ImageNormalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.Tensor(mean).to("cuda").view(-1, 1, 1)
        self.std = torch.Tensor(std).to("cuda").view(-1, 1, 1)

    def forward(self, img):
        img = IMG_STD * img + IMG_MEAN
        return (255*((img - self.mean) / self.std)).flip(dims=(1,))


class PerceptualLossLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.x = x
        return x


pretrained_vgg19 = nn.Sequential(OrderedDict([
        ("conv1_1", nn.Conv2d(3, 64, kernel_size=3, padding=1)),
        ("relu1_1", nn.ReLU(inplace=False)),
        ("conv1_2", nn.Conv2d(64, 64, kernel_size=3, padding=1)),
        ("relu1_2", nn.ReLU(inplace=False)),
        ("pool1", nn.MaxPool2d(kernel_size=2, stride=2)),

        ("conv2_1", nn.Conv2d(64, 128, kernel_size=3, padding=1)),
        ("relu2_1", nn.ReLU(inplace=False)),
        ("conv2_2", nn.Conv2d(128, 128, kernel_size=3, padding=1)),
        ("relu2_2", nn.ReLU(inplace=False)),
        ("pool2", nn.MaxPool2d(kernel_size=2, stride=2)),

        ("conv3_1", nn.Conv2d(128, 256, kernel_size=3, padding=1)),
        ("relu3_1", nn.ReLU(inplace=False)),
        ("conv3_2", nn.Conv2d(256, 256, kernel_size=3, padding=1)),
        ("relu3_2", nn.ReLU(inplace=False)),
        ("conv3_3", nn.Conv2d(256, 256, kernel_size=3, padding=1)),
        ("relu3_3", nn.ReLU(inplace=False)),
        ("conv3_4", nn.Conv2d(256, 256, kernel_size=3, padding=1)),
        ("relu3_4", nn.ReLU(inplace=False)),
        ("pool3", nn.MaxPool2d(kernel_size=2, stride=2)),

        ("conv4_1", nn.Conv2d(256, 512, kernel_size=3, padding=1)),
        ("relu4_1", nn.ReLU(inplace=False)),
        ("conv4_2", nn.Conv2d(512, 512, kernel_size=3, padding=1)),
        ("relu4_2", nn.ReLU(inplace=False)),
        ("conv4_3", nn.Conv2d(512, 512, kernel_size=3, padding=1)),
        ("relu4_3", nn.ReLU(inplace=False)),
        ("conv4_4", nn.Conv2d(512, 512, kernel_size=3, padding=1)),
        ("relu4_4", nn.ReLU(inplace=False)),
        ("pool4", nn.MaxPool2d(kernel_size=2, stride=2)),

        ("conv5_1", nn.Conv2d(512, 512, kernel_size=3, padding=1)),
        ("relu5_1", nn.ReLU(inplace=False)),
        ("conv5_2", nn.Conv2d(512, 512, kernel_size=3, padding=1)),
        ("relu5_2", nn.ReLU(inplace=False)),
        ("conv5_3", nn.Conv2d(512, 512, kernel_size=3, padding=1)),
        ("relu5_3", nn.ReLU(inplace=False)),
        ("conv5_4", nn.Conv2d(512, 512, kernel_size=3, padding=1)),
        ("relu5_4", nn.ReLU(inplace=False)),
        ("pool5", nn.MaxPool2d(kernel_size=2, stride=2)),
]))

pretrained_vgg19.load_state_dict(torch.load(PRETRAINED_VGG19_PATH))
pretrained_vgg19 = pretrained_vgg19.cuda()


def get_perceptual_layers(perceptual_layers: str, pooling: str):
    # model = nn.Sequential(ImageNormalization(mean=VGG_MEAN, std=[1, 1, 1]).to(device))
    model = nn.Sequential(ImageNormalization(mean=VGG_MEAN, std=VGG_STD).to("cuda"))

    perceptual_modules = []
    a, b = 1, 1
    for layer in pretrained_vgg19.children():
        if isinstance(layer, nn.Conv2d):
            name = f"conv{a}_{b}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu{a}_{b}"
            layer = nn.ReLU(inplace=False)
            b += 1
        elif isinstance(layer, nn.MaxPool2d):
            name = f"maxpool{a}"
            a += 1
            b = 1
            if pooling == "max":
                pass
            elif pooling == "avg":
                layer = nn.AvgPool2d(kernel_size=2, stride=2)
            else:
                raise ValueError
        else:
            raise RuntimeError(f"Unrecognized layer: {layer.__class__.__name__}")

        assert name not in model
        model.add_module(name, layer)

        if name in perceptual_layers:
            loss_module = PerceptualLossLayer()
            model.add_module(f"loss_{a}_{b}", loss_module)
            perceptual_modules.append(loss_module)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], PerceptualLossLayer):
            break
    model = model[:i+1]

    assert len(perceptual_layers) == len(perceptual_modules)
    return model, perceptual_modules
