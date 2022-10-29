# coding=utf-8
import numpy as np
import utils as hyper
import scipy.io as sio
import torch
import torch.utils.data as Data
import torchvision.transforms as transforms
from sklearn.preprocessing import MinMaxScaler, Normalizer
import os
from torch.utils.data import Dataset
import torch.nn as nn
import errno
import torch.nn.functional as F
from math import log
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from sklearn.decomposition import PCA
import h5py


def hyperconvert2d(data3d):
    rows, cols, channels = data3d.shape
    data2d = data3d.reshape(rows * cols, channels, order='F')
    return data2d.transpose()


def hyperconvert3d(data2d, rows, cols, channels):
    channels, pixnum = data2d.shape
    data3d = data2d.transpose().reshape(rows, cols, channels, order='F')
    return data3d


# def hypernorm(data2d, flag):
#     normdata = np.zeros(data2d.shape)
#     if flag == "minmax":
#         scaler = MinMaxScaler(feature_range=(0, 1))
#         normdata = scaler.fit_transform(data2d.transpose()).transpose()
#     elif flag == "L2_norm":
#         scaler = Normalizer()
#         normdata = scaler.fit_transform(data2d.transpose()).transpose()
#     else:
#         print("normalization wrong!")
#         exit()
#     return normdata


def hyperminmax(data3d):
    rows, cols, bands = data3d.shape
    data2d = hyperconvert2d(data3d)
    scaler = MinMaxScaler(feature_range=(0, 1))
    normdata = scaler.fit_transform(data2d.transpose()).transpose()
    data3d = hyperconvert3d(normdata, rows, cols, bands)
    return data3d


def calculateAffinties(data3d, sigma=1):
    data2d = hyperconvert2d(data3d)
    data2d = torch.from_numpy(data2d.T)
    dist = torch.cdist(data2d.unsqueeze(dim=0), data2d.unsqueeze(dim=0), p=2)
    dist = dist.squeeze().pow(2)
    affintites = torch.exp(-dist / (2 * (sigma ^ 2)))
    return affintites


def calculatePCA(data3d, k=20):
    rows, cols, bands = data3d.shape
    data2d = data3d.reshape(rows*cols, bands)
    # data3d = data2d.reshape(rows, cols, bands)
    # plt.figure(1)
    # plt.subplot(2, 3, 1)
    # plt.imshow(data3d[:,:,0], label='read_errors')
    # plt.show()
    data2d = data2d.numpy()
    # data2d = hyperconvert2d(data3d)
    # data2d = data2d.T
    pca = PCA(n_components=k)
    data2d = pca.fit_transform(data2d)
    print("PCA OK!")
    return data2d


def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def SandiegoDataset(data_dir):
    datapath = os.path.join(data_dir, "Sandiego.mat")
    gt_path = os.path.join(data_dir, "PlaneGT.mat")
    data = sio.loadmat(datapath)
    # data3d = np.array(data["new_data3D"], dtype=float)
    data3d = np.array(data["Sandiego"], dtype=float)
    data3d = data3d[0:100, 0:100, :]
    remove_bands = np.hstack(
        (range(6), range(32, 35, 1), range(93, 97, 1), range(106, 113), range(152, 166), range(220, 224)))
    data3d = np.delete(data3d, remove_bands, axis=2)
    groundtruthfile = sio.loadmat(gt_path)
    # groundtruthfile = sio.loadmat("Sandiego_II_GT.mat")
    groundtruth = np.array(groundtruthfile["PlaneGT"])
    # groundtruth = np.array(groundtruthfile["new_groundtruth"])
    return data3d, groundtruth


def CriDataset(data_dir):
    datapath = os.path.join(data_dir, "Cri dataset.mat")
    # gt_path = os.path.join(args.data_dir, "PlaneGT.mat")
    data = sio.loadmat(datapath)
    data3d = np.array(data["X"], dtype=float)
    groundtruth = np.array(data["mask"])
    return data3d, groundtruth


def HydiceDataset(data_dir):
    datapath = os.path.join(data_dir, "DataTest_ori.mat")
    gt_path = os.path.join(data_dir, "groundtruth.mat")
    data = sio.loadmat(datapath)
    groundtruth_file = sio.loadmat(gt_path)
    data3d = np.array(data["DataTest_ori"], dtype=float)
    groundtruth = np.array(groundtruth_file["groundtruth"])
    return data3d, groundtruth


def ABU_dataset(data_dir, sub_data_dir):
    datapath = os.path.join(data_dir, sub_data_dir + ".mat")
    data = sio.loadmat(datapath)
    data3d = np.array(data["data"], dtype=float)
    groundtruth = np.array(data["map"])
    return data3d, groundtruth


def WHU_dataset(data_dir, sub_data_dir):
    datapath = os.path.join(data_dir, sub_data_dir + ".mat")
    data = sio.loadmat(datapath)
    data3d = np.array(data[sub_data_dir], dtype=float)
    groundtruth_file = sio.loadmat(os.path.join(data_dir, sub_data_dir + "_gt.mat"))
    groundtruth = np.array(groundtruth_file[sub_data_dir+"_gt"])
    return data3d, groundtruth

def PaviaU_dataset(data_dir):
    datapath = os.path.join(data_dir, "PaviaU.mat")
    gt_path = os.path.join(data_dir, "PaviaU_gt.mat")
    data = sio.loadmat(datapath)
    groundtruth_file = sio.loadmat(gt_path)
    data3d = np.array(data["paviaU"], dtype=float)
    groundtruth = np.array(groundtruth_file["paviaU_gt"])
    return data3d, groundtruth

def Salinas_dataset(data_dir):
    datapath = os.path.join(data_dir, "Salinas_corrected.mat")
    gt_path = os.path.join(data_dir, "Salinas_gt.mat")
    data = sio.loadmat(datapath)
    groundtruth_file = sio.loadmat(gt_path)
    data3d = np.array(data["salinas_corrected"], dtype=float)
    groundtruth = np.array(groundtruth_file["salinas_gt"])
    return data3d, groundtruth

def patch_entropy(data, args):
    data = data.squeeze()
    N = args.outersize ** 2
    uqlabels, uqcount = torch.unique(data, return_counts=True)
    probability = uqcount.float() / N
    log_p = -torch.log2(probability)
    shannonEnt = torch.sum(probability*log_p, dim=0)
    return shannonEnt


def loaddata(dataname, args, sub_data_dir, stride, train=True):
    if dataname == 'Sandiego':
        data3d, groundtruth = SandiegoDataset(args.data_dir)
        # groundtruth = groundtruth.transpose()
    elif dataname == 'Cri':
        data3d, groundtruth = CriDataset(args.data_dir)
        # groundtruth = groundtruth.transpose()
    elif dataname == 'HYDICE':
        data3d, groundtruth = HydiceDataset(args.data_dir)
        # groundtruth = groundtruth.transpose()
    elif dataname == 'ABU':
        data3d, groundtruth = ABU_dataset(args.data_dir, sub_data_dir)
        # groundtruth = groundtruth.transpose()
    elif dataname == 'WHU':
        data3d, groundtruth = WHU_dataset(args.data_dir, sub_data_dir)
        # groundtruth = groundtruth.transpose()
    elif dataname == 'PaviaU':
        data3d, groundtruth = PaviaU_dataset(args.data_dir)
    elif dataname == 'Salinas':
        data3d, groundtruth = Salinas_dataset(args.data_dir)

    else:
        raise NotImplementedError
    output = {}
    output["datasize"] = data3d.shape
    rows, cols, bands = data3d.shape
    # data3d = hypernorm(data3d)
    # data3d = hyperminmax(data3d)
    data3d = torch.from_numpy(data3d.astype(np.float32))
    # if train is False:
    output["affenties"] = data3d
    # data3d = F.normalize(data3d, dim=2)
    data3d = data3d.permute(2, 0, 1).unsqueeze(1)
   # rows, cols = groundtruth.shape
    label = torch.from_numpy(groundtruth).reshape(rows * cols)

    # Generating innner window mask
    mask = torch.ones((args.outersize, args.outersize))
    mask_inner = torch.zeros((args.innersize, args.innersize))
    r = int((args.outersize - args.innersize) / 2)
    mask[r:r + args.innersize, r:r + args.innersize] = mask_inner
    mask = mask.view(-1)
    # mask[int(args.outersize ** 2 / 2)] = 1
    output["mask"] = mask

    # Padding original data
    # pad_size = int(7 / 2)
    # pad = nn.ReplicationPad2d(padding=(pad_size, pad_size, pad_size, pad_size))
    # paddata = pad(data3d)

    # # Generating patch data
    # patchfold = nn.Unfold(kernel_size=7, stride=stride)
    # patchdata = patchfold(paddata).permute(2, 1, 0)
    # data3d_new = patchdata.mean(1).view(rows, cols, bands)
    # data3d_new = data3d_new.permute(2, 0, 1).unsqueeze(1)

    # Padding original data
    pad_size = int(args.outersize / 2)
    pad = nn.ReplicationPad2d(padding=(pad_size, pad_size, pad_size, pad_size))
    paddata = pad(data3d)
    groundtruth = torch.from_numpy(groundtruth.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    padgroundtruth = pad(groundtruth)

    patchfold = nn.Unfold(kernel_size=args.outersize, stride=stride)
    # paddata = pad(data3d_new)
    patchdata = patchfold(paddata).permute(2, 1, 0)
    patchlabel = patchfold(padgroundtruth).squeeze().permute(1, 0)
    output["data"] = patchdata.unsqueeze(2)
    output["label"] = label
    output["patchlabel"] = patchlabel

    # Generating center data
    # centerindex = int(args.outersize**2 / 2)
    # centerdata = patchdata[:, centerindex, :]
    # N, neibor_N, bands = patchdata.shape
    # centerdata_rap = centerdata.unsqueeze(1).expand(N, neibor_N, bands).unsqueeze(2)
    # # self.data = torch.cat([self.data.unsqueeze(2), centerdata_rap], dim=2)
    # self.data2 = centerdata_rap

    # if train is True:
    # explabel = label.unsqueeze(1).expand(len(label), args.outersize ** 2).long()
    # pairlabel = patchlabel == explabel
    # pairlabel = pairlabel.float()
    # output["pairlabel"] = pairlabel
    # mask = output["mask"] > 0
    # pairlabel = pairlabel[:, mask]
    # # anomalyrate = torch.zeros(patchlabel.shape[0])
    # # patchlabel = patchlabel.numpy()
    # # for i in range(patchlabel.shape[0]):
    # #     anomalyrate[i] = patch_entropy(patchlabel[i], args)
    # # a_min, _ = torch.min(anomalyrate, dim=0)
    # # a_max, _ = torch.max(anomalyrate, dim=0)
    # # anomalyrate = (anomalyrate - a_min) / (a_max - a_min)
    # anomalyrate = torch.sum(pairlabel, dim=1) / (args.outersize ** 2 - args.innersize ** 2)
    # # tmp = torch.zeros_like(anomalyrate)
    # # tmp[anomalyrate > 0.7] = 1
    # output["anomalyrate"] = anomalyrate
    return output


class DualWinHyperLoaderh5(Dataset):
    def __init__(self, args, train=True, stride=1):
        self.train = train
        if train is True:
            # data_dir = os.path.join(args.data_dir, args.traindata+".mat")
            # data_dir = os.path.join(args.data_dir, "train")
            # mkdir_if_missing(args.data_dir)
            if args.traindata_sub is None:
                outputpath = os.path.join(args.data_dir, args.traindata +"_{}.h5".format(args.outersize))
            else:
                outputpath = os.path.join(args.data_dir, args.traindata + "_" + args.traindata_sub+"_{}.h5".format(args.outersize))
            dataname = args.traindata

        else:
            # data_dir = os.path.join(args.data_dir, args.testdata+".mat")
            # data_dir = os.path.join(args.data_dir, "test")
            # mkdir_if_missing(args.data_dir)
            if args.testdata_sub is None:
                outputpath = os.path.join(args.data_dir, args.testdata + "_{}.h5".format(args.outersize))
            else:
                outputpath = os.path.join(args.data_dir, args.testdata + "_" + args.testdata_sub + "_{}.h5".format(args.outersize))
            dataname = args.testdata

        if os.path.exists(outputpath):
            self.filepath = outputpath
            self.h5f = None
            with h5py.File(self.filepath, 'r') as h5f:
                self.mask = torch.tensor(h5f['mask'])
                self.datasize = tuple(h5f['datasize'])
                self.lendata = h5f['data'].shape[0]

            if train is True:
                print("Successful loading training data: {}".format(outputpath))
            else:
                self.affenties = torch.tensor(h5py.File(self.filepath, 'r')['affenties'])
                print("Successful loading testing data: {}".format(outputpath))
        else:
            print("Start generating data!")
            if train is True:
                output1 = loaddata(dataname, args, args.traindata_sub, stride, train)
                output2 = loaddata(dataname, args, args.evaldata_sub, stride, train)
                # output1["anomalyrate"] = torch.cat((output1["anomalyrate"], output2["anomalyrate"]), dim=0)
                output1["data"] = torch.cat((output1["data"][:,:,:,:270], output2["data"]), dim=0)
                output1["label"] = torch.cat((output1["label"], output2["label"]), dim=0)
                # output1["pairlabel"] = torch.cat((output1["pairlabel"], output2["pairlabel"]), dim=0)
                output1["patchlabel"] = torch.cat((output1["patchlabel"], output2["patchlabel"]), dim=0)
                with h5py.File(outputpath, 'w') as h5f:
                    h5f['data'] = output1["data"]
                    h5f['mask'] = output1["mask"]
                    h5f['label'] = output1["label"]
                    h5f['affenties'] = output1["affenties"]
                    # h5f['pairlabel'] = output1["pairlabel"]
                    h5f['patchlabel'] = output1["patchlabel"]
                    # h5f['anomalyrate'] = output1["anomalyrate"]
                    h5f['datasize'] = output1["datasize"]
                self.filepath = outputpath
                print("Successful generating training data: {}".format(outputpath))
                with h5py.File(self.filepath, 'r') as h5f:
                    self.mask = torch.tensor(h5f['mask'])
                    self.datasize = tuple(h5f['datasize'])
                    self.lendata = h5f['data'].shape[0]
                    self.affenties = torch.tensor(h5f['affenties'])
                    self.h5f = None
                print("Successful loading training data: {}".format(outputpath))
            else:
                output = loaddata(dataname, args, args.testdata_sub, stride, train)
                with h5py.File(outputpath, 'w') as h5f:
                    h5f['data'] = output["data"]
                    h5f['mask'] = output["mask"]
                    h5f['label'] = output["label"]
                    # h5f['pairlabel'] = output["pairlabel"]
                    h5f['patchlabel'] = output["patchlabel"]
                    # h5f['anomalyrate'] = output["anomalyrate"]
                    h5f['affenties'] = output["affenties"]
                    h5f['datasize'] = output["datasize"]
                self.filepath = outputpath
                print("Successful generating testing data! {}".format(outputpath))
                with h5py.File(self.filepath, 'r') as h5f:
                    self.mask = torch.tensor(h5f['mask'])
                    self.datasize = tuple(h5f['datasize'])
                    self.lendata = h5f['data'].shape[0]
                    self.affenties = torch.tensor(h5f['affenties'])
                    self.h5f = None
                print("Successful loading testing data: {}".format(outputpath))

    def __getitem__(self, idx):
        if self.h5f is None:
            self.h5f = h5py.File(self.filepath, 'r')
        x = self.h5f["data"][idx]
        y = self.h5f["label"][idx]
        # if self.train is True:
        # w = self.h5f["pairlabel"][idx]
        # z = self.h5f["anomalyrate"][idx]
        q = self.h5f["patchlabel"][idx]
        if self.train is True:
            return x,  q
        else:
            return x, y

    def __len__(self):
        return self.lendata




class DualWinHyperLoader(Dataset):
    def __init__(self, args, train=True, stride=1):
        self.train = train
        if train is True:
            # data_dir = os.path.join(args.data_dir, args.traindata+".mat")
            # data_dir = os.path.join(args.data_dir, "train")
            # mkdir_if_missing(args.data_dir)
            if args.traindata_sub is None:
                outputpath = os.path.join(args.data_dir, args.traindata +"_{}.pth.tar".format(args.outersize))
            else:
                outputpath = os.path.join(args.data_dir, args.traindata + "_" + args.traindata_sub+"_{}.pth.tar".format(args.outersize))
            dataname = args.traindata

        else:
            # data_dir = os.path.join(args.data_dir, args.testdata+".mat")
            # data_dir = os.path.join(args.data_dir, "test")
            # mkdir_if_missing(args.data_dir)
            if args.testdata_sub is None:
                outputpath = os.path.join(args.data_dir, args.testdata + "_{}.pth.tar".format(args.outersize))
            else:
                outputpath = os.path.join(args.data_dir, args.testdata + "_" + args.testdata_sub + "_{}.pth.tar".format(args.outersize))
            dataname = args.testdata

        if os.path.exists(outputpath):
            data = torch.load(outputpath)
            self.data = data[0]
            self.mask = data[1]
            self.label = data[2]
            self.datasize = data[-1]
            # self.mask = data[2]
            if train is True:
                self.pairlabel = data[3]
                self.patchlabel = data[4]
                self.anomalyrate = data[5]
                print("Successful loading training data: {}".format(outputpath))
            else:
                self.affenties = data[3]
                print("Successful loading testing data: {}".format(outputpath))
        else:
            print("Start generating data!")
            if dataname == 'Sandiego':
                data3d, groundtruth = SandiegoDataset(args)
                # groundtruth = groundtruth.transpose()
            elif dataname == 'Cri':
                data3d, groundtruth = CriDataset(args)
                # groundtruth = groundtruth.transpose()
            elif dataname == 'HYDICE':
                data3d, groundtruth = HydiceDataset(args)
                # groundtruth = groundtruth.transpose()
            elif dataname == 'ABU':
                data3d, groundtruth = ABU_dataset(args)
                # groundtruth = groundtruth.transpose()
            elif dataname == 'WHU':
                data3d, groundtruth = WHU_dataset(args)
                # groundtruth = groundtruth.transpose()
            elif dataname == 'PaviaU':
                data3d, groundtruth = PaviaU_dataset(args)
            else:
                raise NotImplementedError
            self.datasize = data3d.shape
            # data3d = hypernorm(data3d)
            data3d = hyperminmax(data3d)
            data3d = torch.from_numpy(data3d.astype(np.float32))
            if train is False:
                self.affenties = calculatePCA(data3d)
            # data3d = F.normalize(data3d, dim=2)
            data3d = data3d.permute(2, 0, 1).unsqueeze(1)
            # data3d = F.normalize(data3d, dim=0)
            rows, cols = groundtruth.shape
            label = torch.from_numpy(groundtruth).reshape(rows*cols)

            # Generating innner window mask
            mask = torch.ones((args.outersize, args.outersize))
            mask_inner = torch.zeros((args.innersize, args.innersize))
            r = int((args.outersize - args.innersize) / 2)
            mask[r:r + args.innersize, r:r + args.innersize] = mask_inner
            self.mask = mask.view(-1)
            self.mask[int(args.outersize**2/2)] = 1

            # Padding original data
            pad_size = int(args.outersize/2)
            pad = nn.ReplicationPad2d(padding=(pad_size, pad_size, pad_size, pad_size))
            paddata = pad(data3d)
            groundtruth = torch.from_numpy(groundtruth.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            padgroundtruth = pad(groundtruth)

            # Generating patch data
            patchfold = nn.Unfold(kernel_size=args.outersize, stride=stride)
            patchdata = patchfold(paddata).permute(2, 1, 0)
            patchlabel = patchfold(padgroundtruth).squeeze().permute(1, 0).long()
            self.data = patchdata.unsqueeze(2)
            self.label = label
            self.patchlabel = patchlabel


            # Generating center data
            # centerindex = int(args.outersize**2 / 2)
            # centerdata = patchdata[:, centerindex, :]
            # N, neibor_N, bands = patchdata.shape
            # centerdata_rap = centerdata.unsqueeze(1).expand(N, neibor_N, bands).unsqueeze(2)
            # # self.data = torch.cat([self.data.unsqueeze(2), centerdata_rap], dim=2)
            # self.data2 = centerdata_rap

            if train is True:
                explabel = label.unsqueeze(1).expand(len(label), args.outersize ** 2).long()
                pairlabel = patchlabel == explabel
                pairlabel = pairlabel.float()
                self.pairlabel = pairlabel
                mask = self.mask > 0
                pairlabel = pairlabel[:, mask]
                # anomalyrate = torch.zeros(patchlabel.shape[0])
                # patchlabel = patchlabel.numpy()
                # for i in range(patchlabel.shape[0]):
                #     anomalyrate[i] = patch_entropy(patchlabel[i], args)
                # a_min, _ = torch.min(anomalyrate, dim=0)
                # a_max, _ = torch.max(anomalyrate, dim=0)
                # anomalyrate = (anomalyrate - a_min) / (a_max - a_min)
                anomalyrate = torch.sum(pairlabel, dim=1) / (args.outersize ** 2 - args.innersize ** 2)
                # tmp = torch.zeros_like(anomalyrate)
                # tmp[anomalyrate > 0.7] = 1
                self.anomalyrate = anomalyrate

                torch.save((self.data, self.mask, self.label, self.pairlabel, self.patchlabel, self.anomalyrate, self.datasize), outputpath)
                print("Successful generating training data: {}".format(outputpath))
            else:
                torch.save((self.data, self.mask, self.label, self.affenties, self.datasize), outputpath)
                print("Successful generating testing data! {}".format(outputpath))

    def __getitem__(self, idx):
        x = self.data[idx]
        # x2 = self.data2[idx]
        y = self.label[idx]
        # x = torch.cat([x1, x2], dim=1)
        if self.train is True:
            w = self.pairlabel[idx]
            z = self.anomalyrate[idx]
            # q = self.patchlabel[idx]
            return x,  y, w, z
        else:
            return x, y

    def __len__(self):
        return self.data.shape[0]


class Traindataset(Dataset):
    def __init__(self, args, traindata, traindatasub):
        self.nn = args.outersize**2 - args.innersize**2 + 1
        if args.traindata_sub is None:
            outputpath = os.path.join(args.data_dir, traindata +"_{}.pth.tar".format(args.outersize))
        else:
            outputpath = os.path.join(args.data_dir, traindata + "_" + traindatasub+"_{}.pth.tar".format(args.outersize))
        dataname = traindata
        if os.path.exists(outputpath):
            data = torch.load(outputpath)
            self.data = data[0]
            self.mask = data[1]
            self.label = data[2]
            self.affenties = data[3]
            self.datasize = data[4]
            self.nn = data[5]

            print("Successful loading training data: {}".format(outputpath))

        else:
            print("Start generating data!")
        if dataname == 'Sandiego':
            data3d, groundtruth = SandiegoDataset(args.data_dir)
            # groundtruth = groundtruth.transpose()
        elif dataname == 'Cri':
            data3d, groundtruth = CriDataset(args.data_dir)
            # groundtruth = groundtruth.transpose()
        elif dataname == 'HYDICE':
            data3d, groundtruth = HydiceDataset(args.data_dir)
            # groundtruth = groundtruth.transpose()
        elif dataname == 'ABU':
            data3d, groundtruth = ABU_dataset(args.data_dir, traindatasub)
            # groundtruth = groundtruth.transpose()
        elif dataname == 'WHU':
            data3d, groundtruth = WHU_dataset(args.data_dir, traindatasub)
            # groundtruth = groundtruth.transpose()
        elif dataname == 'PaviaU':
            data3d, groundtruth = PaviaU_dataset(args.data_dir)
        else:
            raise NotImplementedError
        self.datasize = data3d.shape
        rows, cols, bands = data3d.shape
        label = torch.from_numpy(groundtruth).reshape(rows * cols)
        data3d = torch.from_numpy(data3d.astype(np.float32))
        data2d = data3d.reshape(rows*cols, bands)
        self.label = label
        scaler = MinMaxScaler(feature_range=(0, 1))
        data2d = scaler.fit_transform(data2d.numpy())
        data2d = torch.from_numpy(data2d)
        # data2d = data2d[label != 0, :]
        # self.label = label[label != 0]


        # data2d = hyperconvert2d(data3d)
        # tmp = torch.where(label == 0).squeeze()
        # data2d = np.delete(data2d, tmp, axis=1)
        # self.label = np.delete(label, tmp, axis=0)
        # self.label = torch.from_numpy(self.label)

        # data3d = hyperconvert3d(data2d.transpose(), rows, cols, bands)
        # data3d = torch.from_numpy(data3d.astype(np.float32))
        # data2d = torch.from_numpy(data2d.astype(np.float32))
        self.affenties = data2d

        # Generating innner window mask
        mask = torch.ones((args.outersize, args.outersize))
        mask_inner = torch.zeros((args.innersize, args.innersize))
        r = int((args.outersize - args.innersize) / 2)
        mask[r:r + args.innersize, r:r + args.innersize] = mask_inner
        mask = mask.view(-1)
        mask[int(args.outersize ** 2 / 2)] = 1
        self.mask = mask
        self.data = data2d.unsqueeze(1)
        torch.save((self.data, self.mask, self.label, self.affenties, self.datasize, self.nn), outputpath)
        print("Successful generating training data: {}".format(outputpath))

    def __getitem__(self, idx):
        if (self.data.shape[0]-idx)<self.nn:
            idx = self.data.shape[0]-self.nn
        x = self.data[idx:self.nn+idx]
        y = self.label[idx:self.nn+idx]
        # y_c = y[0].expand(y.shape)
        # pairlabel = y == y_c
        # z = pairlabel.float()
        # w = torch.sum(z) / self.nn
        return x, y

    def __len__(self):
        return self.data.shape[0]

