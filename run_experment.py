import argparse
from cheese_main import cheese_experiment
import numpy as np
import warnings
import torch
import torch.multiprocessing as mp
import os
import sys
from checkpoint import Logger


parser = argparse.ArgumentParser(description='Training code - Cheese')
# dataset
# WHU_Hi_HanChuan
parser.add_argument("--data_dir", type=str, default="data", help="path to dataset repository")
parser.add_argument('--traindata', type=str, default="WHU", help='dataset name')
parser.add_argument('--testdata', type=str, default="WHU", help='dataset name')
parser.add_argument('--evaldata', type=str, default="WHU", help='dataset name')
parser.add_argument('--traindata_sub', type=str, default="WHU_Hi_HanChuan", help='only used for WHU dataset')
parser.add_argument('--testdata_sub', type=str, default="WHU_Hi_HanChuan", help='only used for ABU dataset')
parser.add_argument('--evaldata_sub', type=str, default="WHU_Hi_HongHu", help='only used for WHU dataset')
parser.add_argument('--innersize', type=int, default=5, help='inner window size')
parser.add_argument('--outersize', type=int, default=23, help='outer window size')
parser.add_argument('--num_class', default=10, type=int, help='number of class')

# meta learning
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epochs', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--start_episode', default=0, type=int, metavar='N',
                    help='manual iteration number (useful on restarts)')
parser.add_argument('--lr', '--learning_rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[150, 300, 350], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument("--wd", default=1e-5, type=float, help="weight decay")
parser.add_argument('--cos', default=True, help='use cosine lr schedule')
parser.add_argument('--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size')
# VLAD parameters
parser.add_argument('--num_clusters', default=16, type=int, help='number of VLAD prototypes')
parser.add_argument("--feat_dim", default=1024, type=int, help="feature dimension")
parser.add_argument("--hidden_mlp", default=128, type=int,
                    help="hidden layer dimension in projection head")
parser.add_argument("--outputdim", default=256, type=int, help="output feature dimension")

# Transformers parameters #
parser.add_argument("--dim", default=256, type=int,
                    help="Last dimension of output tensor after linear transformation")
parser.add_argument("--dim-head", default=64, type=int, help="Dimension of each head")
parser.add_argument("--depth", default=3, type=int, help="Number of transformer blocks")
parser.add_argument("--heads", default=8, type=int,
                    help="Number of heads in Multi-head Attention layer")
parser.add_argument("--mlp_dim", default=128, type=int,
                    help="Dimension of the MLP (FeedForward) layer")
parser.add_argument("--dropout", default=0.1, type=float, help="Dropout rate between [0,1] ")
parser.add_argument("--emb-dropout", default=0.1, type=float, help="Embedding dropout rate between [0,1] ")
parser.add_argument("--pool", default="cls", type=str, help="cls token pooling or mean pooling")

# General parameters
parser.add_argument("--dump_path", type=str, default="checkpoint",
                    help="experiment dump path for checkpoints and log")
parser.add_argument('--gpu', default=2, type=str, help='GPU id to use.')
parser.add_argument('-j', '--num_work', default=8, type=int, metavar='N',
                    help='number of data loading workers')


cluster_num = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]

def main():
    args = parser.parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # print(torch.cuda.is_available())
    # torch.backends.cudnn.benchmark = True
    sys.stdout = Logger(os.path.join(args.dump_path, 'log.txt'))
    for i in range(1):
        # args.num_clusters = cluster_num[i]
        cheese_experiment(i, args)


if __name__ == '__main__':
    main()
