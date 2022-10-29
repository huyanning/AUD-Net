import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from hard_triplet_loss import HardTripletLoss
import copy
from vit_pytorch import Transformer_layer
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from loss import SupConLoss, FocalLoss



class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten1d(nn.Module):
    def __init__(self, channels):
        self.channels = channels

        super(UnFlatten1d, self).__init__()

    def forward(self, input):
        feature_size = np.int(input.size(1) / self.channels)
        return input.view(input.size(0), self.channels, feature_size)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(in_planes, out_planes, kernel_size=9, stride=stride,
                               padding=4, bias=False)
        self.bn2 = nn.BatchNorm1d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_planes, out_planes, kernel_size=9, stride=1,
                               padding=4, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu(self.bn1(x))
        else:
            out = self.relu(self.bn1(x))

        out = self.relu(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, args, depth, hidden_size=0, widen_factor=1, dropRate=0.0, in_channel=3):
        super().__init__()
        self.args = args
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv1d(in_channel, nChannels[0], kernel_size=9, stride=1,
                               padding=4, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 2, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm1d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.nChannels = nChannels[3]
        self.projection_layer = MLP(nChannels[3], hidden_size, hidden_size)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = 1. / np.sqrt(m.weight.data.size(1))
                m.weight.data.uniform_(-n, n)
                # m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        f_map = self.relu(self.bn1(out))
        # pool_size = f_map.shape[-1]
        # features = F.avg_pool1d(f_map, pool_size)
        # features = out.view(-1, self.nChannels)
        # embeddings = self.projection_layer(features)
        return f_map


class MLP(nn.Module):
    def __init__(self, inputdim, outputdim, hiddendim=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inputdim, hiddendim),
            nn.BatchNorm1d(hiddendim),
            nn.ReLU(inplace=True),
            nn.Linear(hiddendim, outputdim),
            # nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class NetVLAD(nn.Module):
    def __init__(self, num_clusters=16, dim=128, alpha=100.0, normalize_input=True):
        super().__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), stride=1, bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))

    def forward(self, x):
        N, C = x.shape[:2]
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)
        # soft-assigment
        soft_assign = self.conv(x.unsqueeze(2)).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        # (N, C, H, W)->(N, num_clusters, H, W) -> (N, num_clusters, H*W)

        x_flatten = x.view(N, C, -1)
        # calculate residuals to each clusters
        # 减号前面前记为a，后面记为b, residual = a - b
        # a: (N, C, H * W) -> (num_clusters, N, C, H * W) -> (N, num_clusters, C, H * W)
        # b: (num_clusters, C) -> (H * W, num_clusters, C) -> (num_clusters, C, H * W)
        # residual: (N, num_clusters, C, H * W)
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
                   self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        # soft_assign: (N, num_clusters, H * W) -> (N, num_clusters, 1, H * W)
        # (N, num_clusters, C, H * W) * (N, num_clusters, 1, H * W)
        residual *= soft_assign.unsqueeze(2)
        vlad = residual.sum(dim=-1)  # (N, num_clusters, C, H * W) -> (N, num_clusters, C)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten vald: (N, num_clusters, C) -> (N, num_clusters * C)
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad


class CHEESE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = WideResNet(args, depth=10,  hidden_size=args.hidden_mlp,
                     widen_factor=2, dropRate=0.0, in_channel=1).cuda()
        self.args = args
        self.vladnet = NetVLAD(num_clusters=16, dim=128, alpha=100.0, normalize_input=True)
        # self.classifier = MLP(inputdim=16*128, outputdim=2, hiddendim=1024)
        self.classifier = nn.Sequential(
            # nn.Linear(256, 1),
            nn.Linear(128*16, 1),
            # nn.Sigmoid(),
        )
        self.projecter = MLP(inputdim=128*16, outputdim=256, hiddendim=128)
        # self.projecter = nn.Sequential(
        #    nn.Linear(128*16, 256),
        #    nn.ReLU(),
        # )
        #self.fuse_layer = nn.Sequential(
        #    nn.Linear(128*16*2, 256),
        #    nn.ReLU(),
        #)
        self.transformer = Transformer_layer(args)
        # self.triplet_loss = HardTripletLoss(margin=0.1, hardest=True).cuda()
        # self.cross_entropy = nn.CrossEntropyLoss().cuda()
        self.smoothl1loss = nn.SmoothL1Loss().cuda()
        # self.bce_loss = nn.BCEWithLogitsLoss().cuda()
        # self.contrastive_loss = SupConLoss(args)
        self.focal_loss = FocalLoss(alpha=0.9, gamma=2, size_average=True)
        self.mseloss = torch.nn.MSELoss()
        # self.classifier = MLP(dim=2048, projection_size=args.num_class, hidden_size=512)

    def contrastive_loss_fn(self, projection, patchlabel):
        loss = 0
        bs = projection.shape[0]
        # print(projection.shape)
        for i in range(bs):
            loss += self.contrastive_loss(projection[i], patchlabel[i])
        loss = loss / bs
        return loss

    def center_stack(self, projections, targets, mask, args):
        bs, N, dim = projections.shape
        center_index = int(N / 2)
        # center_index = 0
        #innermask = mask<1
        center_label = targets[:, center_index].unsqueeze(1).expand(bs, N)
        pairlabel = center_label == targets
        pairlabel = pairlabel.float()
        # poslabel = pairlabel*2 -1
        mask = mask > 0
        # tmplabel = poslabel[:, mask]
        tmplabel = pairlabel[:, mask]
        anomalyrate = torch.sum(tmplabel, dim=1) / (args.outersize ** 2 - args.innersize ** 2)
        # anomalyrate = torch.sum(tmplabel, dim=1) 
        # anomaly_label = torch.zeros_like(anomalyrate).cuda()
        # anomalyrate[anomalyrate>0.6]=1.

        # centers = projections[:, center_index, :].unsqueeze(1).expand(bs, N, dim)
        #logit = torch.cat((projections, centers), dim=2)
        # logit = projections * centers
        logit = projections
        logit = logit.view(bs*N, -1)
        #logit = self.fuse_layer(logit.view(bs*N, -1))
        return logit, pairlabel, anomalyrate

    def forward(self, inputs, targets, mask):
        # print(pairlabel.shape)
        bs, neighbor_num, channel, bands = inputs.shape
        inputs = F.normalize(inputs, dim=1)
        center_index = int(neighbor_num / 2)
        #center_index = torch.randint(0,neighbor_num, (1,1))[0,0]
        innermask = mask<1
        outermask = mask>0
        centers = inputs[:,innermask,:,:].mean(1)
        centers = centers.unsqueeze(1).expand(bs, neighbor_num, channel, bands)
        # centers = inputs[:, center_index, :, :].unsqueeze(1).expand(bs, neighbor_num, channel, bands)
        inputs = centers*inputs
        # centers = inputs[:, 0, :, :].unsqueeze(1).expand(bs, neighbor_num, channel, bands)
        # logit = torch.cat([inputs, centers], dim=2)
        feature_maps = self.encoder(inputs.view(-1, 1, bands))
        features = self.vladnet(feature_maps)
        logit, pairlabel, anomalyrate = self.center_stack(features.view(bs, neighbor_num, -1), targets, mask, self.args)
        predictions = self.classifier(logit)
        projections = self.projecter(features)
        # projections = F.normalize(projections, dim=1)
        #loss3 = self.contrastive_loss(features.unsqueeze(1), targets.view(bs*neighbor_num, 1))
        # loss3 = self.cross_entropy(projections, targets.view(bs * neighbor_num).long())
        #loss3 = self.triplet_loss(features, targets.view(bs*neighbor_num).float())

        # feature_maps = self.encoder(inputs.view(-1, 1, bands))
        # features = self.vladnet(feature_maps)
        # logit = self.center_stack(features.view(bs, neighbor_num, -1), mask)
        # predictions = self.classifier(logit)
        logit = projections.view(bs, neighbor_num, -1)
        logitouter = logit[:, outermask, :]
        logitinner = logit[:, innermask, :].mean(dim=1).detach()
        logitinner = logit[:, center_index, :].detach()
        x, score = self.transformer(logitouter)
        # predictions = predictions.view(bs, neighbor_num)[:, outermask].view(-1, 1)
        # pairlabel = pairlabel[:, outermask].view(-1, 1)
        loss1 = self.focal_loss(predictions.view(-1, 1), pairlabel.view(-1, 1))
        #loss1 = self.focal_loss(predictions, pairlabel.view(bs*neighbor_num, 1))
        # loss1 = self.bce_loss(predictions.view(-1, 1), pairlabel.view(-1, 1))
        # loss1 = self.cross_entropy(predictions, pairlabel.view(bs * neighbor_num).long())
        # loss2 = self.smoothl1loss(score, anomalyrate.unsqueeze(1))
        # loss2 = self.focal_loss(score, anomalyrate.unsqueeze(1))
        pindex = anomalyrate > 0.5
        # nindex = anomalyrate <= 0.3
        loss2 = self.smoothl1loss(x[pindex, :], logitinner[pindex, :])
        # loss2 = self.smoothl1loss(x, logit.mean(dim=1).detach())
        loss = loss2 + loss1*0.01
        # loss1 = 0
        # loss2 = 0
        loss3 = 0
        loss_list = [loss1, loss2, loss3]
        return loss, loss_list



