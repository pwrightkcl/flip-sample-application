from collections import namedtuple
from pathlib import Path

import torch
import torch.nn
import torch.nn as nn
from torchvision import models as tv


def normalize_tensor(in_feat, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2, dim=1, keepdim=True))
    return in_feat / (norm_factor + eps)


def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2, 3], keepdim=keepdim)


class squeezenet(torch.nn.Module):
    def __init__(self):
        super(squeezenet, self).__init__()
        model = tv.squeezenet1_1(pretrained=False)

        working_dir = Path(__file__).parent.resolve()
        model_path = working_dir / "squeezenet1_1-b8a52dc0.pth"
        if model_path.exists():
            print(f"File does not exist: {str(model_path)}")
            return

        model.load_state_dict(torch.load(str(model_path)))
        pretrained_features = model.features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.slice6 = torch.nn.Sequential()
        self.slice7 = torch.nn.Sequential()
        self.N_slices = 7
        for x in range(2):
            self.slice1.add_module(str(x), pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), pretrained_features[x])
        for x in range(10, 11):
            self.slice5.add_module(str(x), pretrained_features[x])
        for x in range(11, 12):
            self.slice6.add_module(str(x), pretrained_features[x])
        for x in range(12, 13):
            self.slice7.add_module(str(x), pretrained_features[x])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        h = self.slice6(h)
        h_relu6 = h
        h = self.slice7(h)
        h_relu7 = h
        vgg_outputs = namedtuple("SqueezeOutputs", ["relu1", "relu2", "relu3", "relu4", "relu5", "relu6", "relu7"])
        out = vgg_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5, h_relu6, h_relu7)

        return out


# Learned perceptual metric
class LPIPS(nn.Module):
    def __init__(self):

        super().__init__()
        self.scaling_layer = ScalingLayer()

        net_type = squeezenet
        self.chns = [64, 128, 256, 384, 384, 512, 512]
        self.L = len(self.chns)

        self.net = net_type()

        self.lin0 = NetLinLayer(self.chns[0], use_dropout=True)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=True)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=True)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=True)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=True)
        self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        self.lin5 = NetLinLayer(self.chns[5], use_dropout=True)
        self.lin6 = NetLinLayer(self.chns[6], use_dropout=True)
        self.lins += [self.lin5, self.lin6]
        self.lins = nn.ModuleList(self.lins)

        working_dir = Path(__file__).parent.resolve()
        model_path = working_dir / "squeeze.pth"
        if model_path.exists():
            print(f"File does not exist: {str(model_path)}")
            return

        self.load_state_dict(torch.load(str(model_path), map_location="cpu"), strict=False)

        self.eval()

    def forward(self, in0, in1):
        # v0.0 - original release had a bug, where input was not scaled
        in0_input, in1_input = self.scaling_layer(in0), self.scaling_layer(in1)
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}

        for kk in range(self.L):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [spatial_average(self.lins[kk](diffs[kk]), keepdim=True) for kk in range(self.L)]

        val = 0
        for l in range(self.L):
            val += res[l]

        return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer("shift", torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None])
        self.register_buffer("scale", torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """A single linear layer which does a 1x1 conv"""

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        layers = (
            [
                nn.Dropout(),
            ]
            if (use_dropout)
            else []
        )
        layers += [
            nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

perceptual = LPIPS()