import torch
import torch.nn as nn
import torch.nn.functional as F

import InfoNCE_Loss
from DropBlock import DropBlock2D


class Bottleneck(nn.Module):
    def __init__(self, encode_num, inplanes, planes, stride=1, expansion=4):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * expansion, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(planes * expansion)
        self.relu = nn.ReLU(True)
        if encode_num == 0:
            self.dropblock = DropBlock2D(drop_prob=0.02, block_size=7)
        elif encode_num == 3:
            self.dropblock = DropBlock2D(drop_prob=0.1, block_size=3)
        else:
            self.dropblock = DropBlock2D(drop_prob=0.05, block_size=5)
        if stride != 1 or inplanes != expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, expansion * planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(expansion * planes)
            )

    def forward(self, x):
        shortcut = self.shortcut(x) if hasattr(self, "shortcut") else x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x += shortcut
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, opt, encode_num, block=Bottleneck, input_dims=3):
        super(ResNet, self).__init__()
        self.expansion = 4
        self.opt = opt
        self.block = block
        self.inplanes = [64, 256, 512, 1024]
        self.channels = [64, 128, 256, 512]
        self.num_blocks = [3, 4, 6, 3]
        self.patch_size = 48
        self.overlap = 2
        i = encode_num
        self.encode_num = encode_num
        if opt.dataset == 'cifar10':
            self.layer0 = nn.Sequential(
                nn.Conv2d(input_dims, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True)
            )
        elif opt.dataset == 'stl10':
            self.layer0 = nn.Sequential(
                nn.Conv2d(input_dims, 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
            )
        self.model = nn.Sequential()
        if i == 0:
            self.model.add_module("first", self.layer0)
            self.model.add_module(
                "conv {}".format(i),
                self._make_layer(
                    i,
                    self.block,
                    self.inplanes[i],
                    self.channels[i],
                    self.num_blocks[i],
                    stride=1
                )
            )
        else:
            self.model.add_module(
                "conv {}".format(i),
                self._make_layer(
                    i,
                    self.block,
                    self.inplanes[i],
                    self.channels[i],
                    self.num_blocks[i],
                    stride=2
                )
            )
        self.last = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.pro_head = nn.Sequential(nn.Linear(self.channels[i] * self.expansion, 512, bias=False), nn.BatchNorm1d(512),
                                      nn.ReLU(inplace=True), nn.Linear(512, opt.feature_dim, bias=True))


    def _make_layer(self, encode_num, block, inplanes, planes, blocks, stride, expansion=4):
        layers = []
        layers.append(block(encode_num, inplanes, planes, stride, expansion))
        for i in range(1, blocks):
            layers.append(block(encode_num, planes * expansion, planes, expansion=expansion))
        return nn.Sequential(*layers)

    def forward(self, x_1, x_2, opt):
        # if self.encode_num == 0:
        #     x_patch = (
        #         x_patch.unfold(2, self.patch_size, self.patch_size // self.overlap)
        #             .unfold(3, self.patch_size, self.patch_size // self.overlap)
        #             .permute(0, 2, 3, 1, 4, 5)
        #     )
        #     n_patches_x = x_patch.shape[1]
        #     n_patches_y = x_patch.shape[2]
        #     x_patch = x_patch.reshape(
        #         x_patch.shape[0] * x_patch.shape[1] * x_patch.shape[2], x_patch.shape[3], x_patch.shape[4], x_patch.shape[5]
        #     )
        # z_patch = self.model(x_patch)
        # out_patch = F.adaptive_avg_pool2d(z_patch, 1)
        # out_patch = out_patch.reshape(-1, 3, 3, out_patch.shape[1])
        # out_patch = out_patch.permute(0, 3, 1, 2).contiguous()
        # z_patch = F.adaptive_avg_pool2d(z_patch, 1)
        # feature_patch = torch.flatten(z_patch, start_dim=1)
        # out_patch = self.pro_head(feature_patch)
        # out_patch = out_patch.reshape(-1, 7, 7, out_patch.shape[1])
        # out_patch = out_patch.permute(0, 3, 1, 2).contiguous()
        # loss_patch = self.loss(out_patch, out_patch)
        x_1 = self.model(x_1)
        z_1 = self.last(x_1)
        feature_1 = torch.flatten(z_1, start_dim=1)
        out_1 = self.pro_head(feature_1)
        x_2 = self.model(x_2)
        z_2 = self.last(x_2)
        feature_2 = torch.flatten(z_2, start_dim=1)
        out_2 = self.pro_head(feature_2)
        loss_sim = contrast_loss(out_1, out_2, opt)
        # loss = loss_patch + loss_sim
        return x_1, F.normalize(feature_1, dim=-1), F.normalize(out_1, dim=-1),\
               x_2, F.normalize(feature_2, dim=-1), F.normalize(out_2, dim=-1), loss_sim


def contrast_loss(out_1, out_2, opt):
    out = torch.cat([out_1, out_2], dim=0)
    size = out_1.size()[0]
    n = size * 2
    s = torch.matmul(out, torch.transpose(out, 0, 1)) / (
        opt.tem * torch.matmul(torch.norm(out, dim=1).unsqueeze(1), torch.norm(out, dim=1).unsqueeze(0))
    )
    a = torch.diag(s)
    l = -(s.exp() / (s.exp().sum(dim=0)-a.exp())).log()
    l1 = torch.diag(l[0: size, size: 2 * size])
    l2 = torch.diag(l[size: 2 * size, 0: size])
    loss = ((l1 + l2).sum())/n
    return loss


class FullModel(nn.Module):
    def __init__(self, opt):
        super(FullModel, self).__init__()
        self.encode_num = range(4)
        self.dim = [256, 512, 1024, 2048]
        self.encoder = nn.ModuleList([])
        self.pro_head = nn.ModuleList([])
        self.opt = opt
        for i in self.encode_num:
            self.encoder.append(ResNet(opt, i))
            self.pro_head.append(
                nn.Sequential(nn.Linear(self.dim[i], 512, bias=False), nn.BatchNorm1d(512),
                              nn.ReLU(inplace=True), nn.Linear(512, opt.feature_dim, bias=True))
            )
        print(self.encoder)

    def forward(self, x_1, x_2, num_GPU, opt, n=4):
        input_1 = x_1
        input_2 = x_2
        # input_patch = x_1
        if self.opt.device.type != "cpu":
            cur_device = x_1.get_device()
        else:
            cur_device = self.opt.device
        loss = torch.zeros(1, 4, device=cur_device)
        for idx, module in enumerate(self.encoder[: n+1]):
            z_1, feature_1, cur_out_1, z_2, feature_2, cur_out_2, cur_loss = module(input_1, input_2, opt)
            # Detach z to make sure no gradients are flowing in between modules
            input_1 = z_1.detach()
            input_2 = z_2.detach()
            # input_patch = z_patch.detach()
            loss[:, idx] = cur_loss

        return feature_1, feature_2, cur_out_1, cur_out_2, loss


