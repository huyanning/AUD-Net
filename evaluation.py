# coding=utf-8
import torch.optim as optim
import torch
import numpy as np
from memory import Memory
import os
from utils import AverageLoss, evaluator, get_Mahalanobis_convariance, mahalanobis, adjust_learning_rate, adjust_learning_rate_poly
import time
from checkpoint import load_checkpoint, save_checkpoint, copy_state_dict, Logger
import sys
import torch.nn.functional as F
from torchvision import datasets, transforms
from datasets import DualWinHyperLoaderh5
import torch.utils.data as Data
from sklearn.metrics import roc_auc_score
from test import anomaly_show, scatter_plot
from sklearn.preprocessing import MinMaxScaler
from model import CHEESE
from utils import patch2image
import matplotlib.pyplot as plt
import scipy.io as sio
import os.path as osp
import argparse
# from cheese_main import cheese_experiment
import numpy as np
import warnings
import torch
import torch.multiprocessing as mp
import os
import sys
from checkpoint import Logger
from sklearn.neighbors import KNeighborsRegressor


# parser = argparse.ArgumentParser(description='Training code - Cheese')
# # dataset
# parser.add_argument("--data_dir", type=str, default="data", help="path to dataset repository")
# parser.add_argument('--traindata', type=str, default="WHU", help='dataset name')
# parser.add_argument('--testdata', type=str, default="Sandiego", help='dataset name')
# parser.add_argument('--evaldata', type=str, default="WHU", help='dataset name')
# parser.add_argument('--traindata_sub', type=str, default="WHU_Hi_LongKou", help='only used for WHU dataset')
# parser.add_argument('--testdata_sub', type=str, default=None, help='only used for ABU dataset')
# parser.add_argument('--evaldata_sub', type=str, default="WHU_Hi_HongHu", help='only used for WHU dataset')
# parser.add_argument('--innersize', type=int, default=1, help='inner window size')
# parser.add_argument('--outersize', type=int, default=5, help='outer window size')
# parser.add_argument('--num_class', default=10, type=int, help='number of class')
#
# # meta learning
# parser.add_argument('--epochs', default=20, type=int, metavar='N',
#                     help='number of total epochs to run')
# parser.add_argument('--start_epochs', default=0, type=int, metavar='N',
#                     help='manual epoch number (useful on restarts)')
# parser.add_argument('--start_episode', default=0, type=int, metavar='N',
#                     help='manual iteration number (useful on restarts)')
# parser.add_argument('--lr', '--learning_rate', default=0.001, type=float,
#                     metavar='LR', help='initial learning rate', dest='lr')
# parser.add_argument('--schedule', default=[150, 300, 350], nargs='*', type=int,
#                     help='learning rate schedule (when to drop lr by 10x)')
# parser.add_argument("--wd", default=1e-5, type=float, help="weight decay")
# parser.add_argument('--cos', default=True, help='use cosine lr schedule')
# parser.add_argument('--batch-size', default=256, type=int,
#                     metavar='N', help='mini-batch size')
# # VLAD parameters
# parser.add_argument('--num_clusters', default=16, type=int, help='number of VLAD prototypes')
# parser.add_argument("--feat_dim", default=1024, type=int, help="feature dimension")
# parser.add_argument("--hidden_mlp", default=128, type=int,
#                     help="hidden layer dimension in projection head")
# parser.add_argument("--outputdim", default=256, type=int, help="output feature dimension")
#
# # Transformers parameters #
# parser.add_argument("--dim", default=2048, type=int,
#                     help="Last dimension of output tensor after linear transformation")
# parser.add_argument("--dim-head", default=64, type=int, help="Dimension of each head")
# parser.add_argument("--depth", default=3, type=int, help="Number of transformer blocks")
# parser.add_argument("--heads", default=8, type=int,
#                     help="Number of heads in Multi-head Attention layer")
# parser.add_argument("--mlp_dim", default=1024, type=int,
#                     help="Dimension of the MLP (FeedForward) layer")
# parser.add_argument("--dropout", default=0.1, type=float, help="Dropout rate between [0,1] ")
# parser.add_argument("--emb-dropout", default=0.1, type=float, help="Embedding dropout rate between [0,1] ")
# parser.add_argument("--pool", default="cls", type=str, help="cls token pooling or mean pooling")
#
# # General parameters
# parser.add_argument("--dump_path", type=str, default="checkpoint",
#                     help="experiment dump path for checkpoints and log")
# parser.add_argument('--gpu', default=0, type=str, help='GPU id to use.')
# parser.add_argument('-j', '--num_work', default=8, type=int, metavar='N',
#                     help='number of data loading workers')
#
#
# cluster_num = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]


def get_score(projections, mask):
    bs, N, dim = projections.shape
    # tmp = mask.sum(dim=0)
    # mask = mask.unsqueeze(0).unsqueeze(2).expand(bs, N, dim)
    # centers = (projections*mask).sum(dim=1, keepdim=True).expand(bs, N, dim) / tmp
    center_index = int(N / 2)
    centers = projections[:, center_index, :].unsqueeze(1).expand(bs, N, dim)
    projections = F.normalize(projections, dim=2)
    centers = F.normalize(centers, dim=2)
    logit = (projections * centers).sum(dim=2)
    return 1 - ((1 + logit) / 2)


def get_global_score(projections):
    projections = projections.mean(dim=1)
    N, dim = projections.shape
    center = projections.mean(dim=0, keepdim=True).expand(N, dim)
    projections = F.normalize(projections, dim=1)
    center = F.normalize(center, dim=1)
    logit = (projections * center).sum(dim=1)
    return 1 - (1 + logit) / 2


def get_affinties_score(predictions, affinties):
    predictions = torch.from_numpy(predictions).squeeze()
    predictions = predictions.unsqueeze(0).expand(affinties.shape)
    predictions = (predictions * affinties).sum(dim=1)
    return predictions.numpy()


def get_KNN_score(X, predictions, K=30):
    rows, cols, bands = X.shape
    X = X.reshape(rows*cols, bands)
    knr = KNeighborsRegressor(n_neighbors=K, weights='uniform')
    knr.fit(X, predictions)
    score = knr.predict(X)
    return score


def center_stack(projections):
    bs, N, dim = projections.shape
    center_index = int(N / 2)
    centers = projections[:, center_index, :].unsqueeze(1).expand(bs, N, dim)
    # logit = projections * centers
    loggit = projections
    return logit


def detector(learner, dataloader, count, mask, args):
    print("start detecting...")
    learner.eval()
    features_list = []
    label_list = []
    prediction_list = []
    score_list = []
    projections_list = []
    innermask = mask < 1
    outermask = mask > 0
    for i, (inputs, labels) in enumerate(dataloader):
        bs, neighbor_num, channels, bands = inputs.shape
        with torch.no_grad():
            inputs = inputs.cuda(args.gpu)
            labels = labels.cuda(args.gpu)
            inputs = F.normalize(inputs, dim=1)
            center_index = int(neighbor_num / 2)
            centers = inputs[:, innermask, :, :].mean(1)
            centers = centers.unsqueeze(1).expand(bs, neighbor_num, channels, bands)
            # centers = inputs[:, 0, :, :].unsqueeze(1).expand(bs, neighbor_num, channels, bands)
            inputs = centers * inputs
            feature_maps = learner.encoder(inputs.view(-1, 1, bands))
            features = learner.vladnet(feature_maps)
            projections = learner.projecter(features)
            # logit = center_stack(features.view(bs, neighbor_num, -1))
            # logit = logit.view(bs*N, -1)
            # predictions = learner.classifier(features).view(bs, neighbor_num)
            predictions = learner.classifier(features)
            logitinner = projections.view(bs, neighbor_num, -1)[:, center_index, :].detach()
            # logitinner = projections.view(bs, neighbor_num, -1)[:, innermask, :].mean(dim=1).detach()
            logit = projections.view(bs, neighbor_num, -1)[:, outermask, :]
            x, anomaly_score = learner.transformer(logit)
            anomaly_score = ((x - logitinner)**2).sum(dim=1)
            label_list.append(labels)
            prediction_list.append(predictions)
            score_list.append(anomaly_score)
        if i % 1000 == 0:
            print("Currnt mini-batches: {}".format(i))

    rows, cols, bands = dataloader.dataset.datasize
    labels = torch.cat(label_list, dim=0).cpu().detach().numpy()
    predictions = torch.cat(prediction_list, dim=0).cpu().detach().numpy()
    score = torch.cat(score_list, dim=0).cpu().detach().numpy()
    w = args.outersize
    predictions = predictions.reshape(rows * cols, w * w)
    
    
    knn_score = get_KNN_score(dataloader.dataset.affenties, score)
    weight = np.ones_like(score)
    img_pre = patch2image(predictions, weight, [rows, cols], args)
    img = patch2image(predictions, 1-knn_score, [rows, cols], args)
    predictions_mean = predictions[:, outermask.cpu().numpy()].mean(axis=1)
    predictions = img_pre.reshape(rows * cols)
    weighted_predictions = img.reshape(rows * cols)
    # new_predictions = get_affinties_score(predictions, dataloader.dataset.affenties)
    combine_predictions = predictions_mean * knn_score.squeeze()
    # combine_weighted_predictions = combine_predictions * score.squeeze()


    # img_combine = patch2image(combine_predictions, score, [rows, cols], args)
    # combine_predictions = np.sum(combine_predictions, axis=1) / args.outersize ** 2
    # combine_weighted_predictions = img_combine.reshape(rows * cols)
    # for i in range(len(test_features)):
    #     m_distance[i] = mahalanobis(test_features[i], mean_train, inv_cov_train)
    # features = torch.cat(feature_list, dim=0).cpu().detach().numpy()
    # labels = torch.cat(label_list, dim=1).cpu().detach().numpy()
    # scatter_plot(encodes1, label, epoch, args)
    # scatter_show(y_train, targets, encodes2, 2, epoch)
    # histogram_modified(errors[y_train == 1], errors[y_train==0], args, epoch)
    # show(targets, embeddings1, epoch, args)
    knn_score = knn_score.squeeze()
    score_norm = (knn_score - knn_score.min(axis=0)) / (knn_score.max(axis=0) - knn_score.min(axis=0))
    predictions_norm = (predictions - predictions.min(axis=0)) / (predictions.max(axis=0) - predictions.min(axis=0))
    fuse_score = 0.6*predictions_norm + 0.4*(1-score_norm)
    roauc = np.zeros(7)

    roauc[0], pr_auc_norm, pr_auc_anom = \
        evaluator(knn_score, labels, "featuresum", args)
    roauc[1], pr_auc_norm, pr_auc_anom = \
        evaluator(score, labels, "score", args)
    roauc[2], pr_auc_norm_p, pr_auc_anom_p = \
        evaluator(1-predictions, labels, "predictions", args)
    roauc[3], pr_auc_norm_w, pr_auc_anom_w = \
        evaluator(1-weighted_predictions, labels, "weighted_predictions", args)
    roauc[4], pr_auc_norm_g, pr_auc_anom_g = \
        evaluator(1-predictions_mean, labels, "predictions_mean", args)
    roauc[5], pr_auc_norm_cw, pr_auc_anom_cw = \
        evaluator(1-fuse_score, labels, "fuse_score", args)
    roauc[6], pr_auc_norm_cw, pr_auc_anom_cw = \
        evaluator(1-combine_predictions, labels, "fuse_score", args)
    anomaly_show(score, 1 - predictions, 1- weighted_predictions, knn_score, 1-predictions_mean,
                 1 - fuse_score,
                 count, args)
    return roauc


def run_test(learner, epoch, args):
    test_data = DualWinHyperLoaderh5(args, train=False, stride=1)
    args.rows, args.cols, args.bands = test_data.datasize
    print("Testing data size: {}".format(test_data.datasize))
    args.num_neighbors = args.outersize**2 - args.innersize**2
    test_loader = Data.DataLoader(dataset=test_data,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=args.num_work)

    print("Current test data: {}-{}".format(args.testdata, args.testdata_sub))
    roauc=detector(learner, test_loader, epoch, test_data.mask.cuda(args.gpu), args)
    return roauc


def test(epoch, args):
    dataname_list = ["ABU", "ABU", "ABU", "ABU", "ABU", "ABU", "ABU", "ABU", "ABU", "ABU",
                     "ABU", "ABU", "ABU"]
    subdataname_list = ["abu-airport-1", "abu-airport-2", "abu-airport-3", "abu-airport-4",
                        "abu-beach-1", "abu-beach-2", "abu-beach-3", "abu-beach-4", "abu-urban-1", "abu-urban-2",
                        "abu-urban-3", "abu-urban-4", "abu-urban-5"]
    # args = parser.parse_args()
    sys.stdout = Logger(os.path.join(args.dump_path, 'evaluation_log.txt'))
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    args.run_times = 0
    running_info = '{}-cheese{}-{}-{}'.format(args.run_times, args.num_clusters, args.traindata, args.traindata_sub)
    model_path = os.path.join(args.dump_path, running_info)
    learner = CHEESE(args).cuda(args.gpu)
    # learner = torch.nn.DataParallel(learner, device_ids=range(torch.cuda.device_count())).cuda()
    optimizer = optim.Adam(learner.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)
    # optimizer = optim.SGD(learner.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    load_checkpoint(learner, optimizer, model_path, args, epoch)
    start_epoch = args.start_epoch
    print("The model has been trained {} epochs".format(start_epoch))
    auroc = np.zeros((7, 13))
    for i in range(13):
        args.testdata = dataname_list[i]
        args.testdata_sub = subdataname_list[i]
        auroc[:, i] = run_test(learner, epoch, args)
        print(auroc[:,i])
    print("Final mean auroc: {}".format(auroc.mean(axis=1)))


# if __name__ == '__main__':
#     test(2)
