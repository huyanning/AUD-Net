# coding=utf-8
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from memory import Memory
from loss import info_nce_loss


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


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        def Basic(Input, Output):
            return torch.nn.Sequential(
                torch.nn.Conv1d(in_channels=Input, out_channels=Output, kernel_size=9, stride=1, padding=4),
                torch.nn.BatchNorm1d(Output),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv1d(in_channels=Output, out_channels=Output, kernel_size=9, stride=1, padding=4),
                torch.nn.BatchNorm1d(Output),
                torch.nn.ReLU(inplace=False)
            )

        def Basic_(Input, Output):
            return torch.nn.Sequential(
                torch.nn.Conv1d(in_channels=Input, out_channels=Output, kernel_size=9, stride=1, padding=4),
                torch.nn.BatchNorm1d(Output),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv1d(in_channels=Output, out_channels=Output, kernel_size=9, stride=1, padding=4),
            )

        self.moduleConv1 = Basic(1, 64)
        self.modulePool1 = torch.nn.MaxPool1d(kernel_size=2, stride=2)

        self.moduleConv2 = Basic(64, 128)
        self.modulePool2 = torch.nn.MaxPool1d(kernel_size=2, stride=2)

        self.moduleConv3 = Basic(128, 256)
        self.modulePool3 = torch.nn.MaxPool1d(kernel_size=2, stride=2)

        self.moduleConv4 = Basic_(256, 512)
        self.moduleBatchNorm = torch.nn.BatchNorm1d(512)
        self.moduleReLU = torch.nn.ReLU(inplace=False)
        self.fc = nn.Sequential(
            Flatten(),
            torch.nn.Linear(23*512, 1024),
            # torch.nn.ReLU(inplace=False),   
        )

    def forward(self, x):
        out = self.moduleConv1(x)
        out = self.modulePool1(out)

        out = self.moduleConv2(out)
        out = self.modulePool2(out)

        out = self.moduleConv3(out)
        out = self.modulePool3(out)

        out = self.moduleConv4(out)
        # out = self.moduleBatchNorm(out)
        out = self.moduleReLU(out)
        # print(out.shape)
        out = self.fc(out)
        # features = F.avg_pool1d(out, 23).squeeze()
        
        return out


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        def Basic(Input, Output):
            return torch.nn.Sequential(
                torch.nn.Conv1d(in_channels=Input, out_channels=Output, kernel_size=9, stride=1, padding=4),
                torch.nn.BatchNorm1d(Output),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv1d(in_channels=Output, out_channels=Output, kernel_size=9, stride=1, padding=4),
                torch.nn.BatchNorm1d(Output),
                torch.nn.ReLU(inplace=False)
            )

        def Gen(Input, Output, nc):
            return torch.nn.Sequential(
                torch.nn.ConvTranspose1d(in_channels=Input, out_channels=nc, kernel_size=6, stride=1, padding=0),
                torch.nn.BatchNorm1d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv1d(in_channels=nc, out_channels=nc, kernel_size=9, stride=1, padding=4),
                torch.nn.BatchNorm1d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv1d(in_channels=nc, out_channels=Output, kernel_size=9, stride=1, padding=4),
                torch.nn.Tanh()
            )

        def Upsample(nc, Output, stride=1, outpadding=1):
            return torch.nn.Sequential(
                torch.nn.ConvTranspose1d(in_channels=nc, out_channels=Output, kernel_size=9, stride=stride, padding=4,
                                         output_padding=outpadding),
                torch.nn.BatchNorm1d(Output),
                torch.nn.ReLU(inplace=False)
            )

        self.moduleConv = Basic(512, 256)
        self.moduleUpsample4 = Upsample(256, 256, 2, 1)

        self.moduleDeconv3 = Basic(256, 128)
        self.moduleUpsample3 = Upsample(128, 128, 2,1)

        self.moduleDeconv2 = Basic(128, 64)
        self.moduleUpsample2 = Upsample(64, 64, 2,1)

        self.moduleDeconv1 = Gen(64, 1, 64)
        self. tfc = torch.nn.Sequential(
            torch.nn.Linear(2048, 23*512),
            torch.nn.ReLU(inplace=True),
            UnFlatten1d(512),
            )
    def forward(self, x):
        # print(x.shape)
        x = self.tfc(x)
        tensorConv = self.moduleConv(x)
        tensorUpsample4 = self.moduleUpsample4(tensorConv)
        # print(tensorUpsample4.shape)
        tensorDeconv3 = self.moduleDeconv3(tensorUpsample4)
        tensorUpsample3 = self.moduleUpsample3(tensorDeconv3)
        # print(tensorUpsample3.shape)

        tensorDeconv2 = self.moduleDeconv2(tensorUpsample3)
        tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)
        # print(tensorUpsample2.shape)

        output = self.moduleDeconv1(tensorUpsample2)

        return output


class Cluster_layer(nn.Module):
    def __init__(self, input_size, hidden_size, prototype_num):
        super().__init__()
        self.net = nn.Sequential(

            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, prototype_num),
            # nn.Sigmoid()
        )

    def forward(self, x):
        # out = F.avg_pool2d(x, 8).view(-1, 256)
        return self.net(x)


class MemAE(torch.nn.Module):
    def __init__(self, args):
        super(MemAE, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()
        self.cluster_layer = Cluster_layer(args.feat_dim, args.hidden_mlp, args.num_clusters)
        self.memory = Memory(args, memory_init=None)
        # self.info_nce_loss = info_nce_loss()
        self.args = args
        self.crossentropy = nn.CrossEntropyLoss()

    def forward(self, inputs, index, selflabels, epoch):
        bs = int(inputs.shape[0] / 2)
        encodes = self.encoder(inputs)
        clusters = self.cluster_layer(encodes)
        cluster_loss = self.crossentropy(clusters, selflabels[index])
        read_item = self.memory.read(clusters)
        decodes = self.decoder(torch.cat((read_item, encodes), dim=1))
        # decodes = self.decoder(read_item)
        reconstruction_loss = F.mse_loss(decodes, inputs, reduction='mean')

        if epoch >= self.args.warmup_epochs:
            
            # encodes = F.normalize(encodes, dim=1)
            read_mse = F.mse_loss(encodes, read_item.detach(), reduction='mean')
            # loss = reconstruction_loss 
            # read_mse=0

            loss = reconstruction_loss + cluster_loss

        else:
            # decodes = self.decoder(encodes)
            # reconstruction_loss = F.mse_loss(decodes, inputs)
            loss = reconstruction_loss + cluster_loss*0.01
            read_mse = 0
            #loss = reconstruction_loss
            
        loss_dic = [reconstruction_loss, cluster_loss, read_mse]
        return loss, loss_dic

