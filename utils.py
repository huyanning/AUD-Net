import os
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import math
from logging import getLogger
import pickle
import os
import pandas as pd
import time
from datetime import timedelta
import logging
import torch
import torch.distributed as dist
import warnings
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os.path as osp
logger = getLogger()


def save_roc_pr_curve_data(scores, labels, file_path):
    preds = scores.squeeze()
    truth = labels.squeeze()

    # scores_pos = scores[labels == 1]
    # scores_neg = scores[labels != 1]

    # truth = np.concatenate((np.zeros_like(scores_neg), np.ones_like(scores_pos)))
    # preds = np.concatenate((scores_neg, scores_pos))
    fpr, tpr, roc_thresholds = roc_curve(truth, preds)
    roc_auc = auc(fpr, tpr)

    # pr curve where "normal" is the positive class
    precision_norm, recall_norm, pr_thresholds_norm = precision_recall_curve(truth, preds)
    pr_auc_norm = auc(recall_norm, precision_norm)

    # pr curve where "anomaly" is the positive class
    precision_anom, recall_anom, pr_thresholds_anom = precision_recall_curve(truth, -preds, pos_label=0)
    pr_auc_anom = auc(recall_anom, precision_anom)

    np.savez_compressed(file_path,
                        preds=preds, truth=truth,
                        fpr=fpr, tpr=tpr, roc_thresholds=roc_thresholds, roc_auc=roc_auc,
                        precision_norm=precision_norm, recall_norm=recall_norm,
                        pr_thresholds_norm=pr_thresholds_norm, pr_auc_norm=pr_auc_norm,
                        precision_anom=precision_anom, recall_anom=recall_anom,
                        pr_thresholds_anom=pr_thresholds_anom, pr_auc_anom=pr_auc_anom)


def init_distributed_mode(args):
    """
    Initialize the following variables:
    - word_size
    - rank
    """
    if args.gpu is not None:
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    else:
        # multi-GPU job (local or multi-node) - jobs started with torch.distributed.launch
        # read environment variables
        os.environ['MASTER_ADDR'] = '172.17.0.2'
        os.environ['MASTER_PORT'] = '29555'
        args.rank = args.local_rank
        args.world_size = args.nprocs

        # prepare distributed
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=args.world_size,
            rank=args.rank,
        )

        # set cuda device
        args.gpu_to_work_on = args.rank % torch.cuda.device_count()
        torch.cuda.set_device(args.gpu_to_work_on)
        return


def get_class_name_from_index(index, dataset_name):
    ind_to_name = {
        'cifar10': ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'),
        'cifar100': ('aquatic mammals', 'fish', 'flowers', 'food containers', 'fruit and vegetables',
                     'household electrical devices', 'household furniture', 'insects', 'large carnivores',
                     'large man-made outdoor things', 'large natural outdoor scenes', 'large omnivores and herbivores',
                     'medium-sized mammals', 'non-insect invertebrates', 'people', 'reptiles', 'small mammals', 'trees',
                     'vehicles 1', 'vehicles 2'),
        'fashion-mnist': ('t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag',
                          'ankle-boot'),
        'cats-vs-dogs': ('cat', 'dog'),
        'mnist':('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'),
        'svhn':('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    }

    return ind_to_name[dataset_name][index]


def save_result(auroc, pr_in, pr_out, result):
    index = np.argmax(auroc)
    final_auroc = auroc(index)
    final_pr_in = pr_in(index)
    final_pr_out= pr_out(index)
    result[0] = final_auroc
    result[1] = final_pr_in
    result[2] = final_pr_out
    return result


def evaluator(predict, target, flag, args):
    # run_times = args.run_times
    # dataset_name = args.testdata
    predict_pos = predict[target == 1]
    predict_neg = predict[target != 1]
    # calculate AUC
    truth = np.concatenate((np.zeros_like(predict_neg), np.ones_like(predict_pos)))
    predict = np.concatenate((predict_neg, predict_pos))
    fpr, tpr, roc_thresholds = roc_curve(truth, predict)
    roc_auc = auc(fpr, tpr)

    # PR curve where "normal" is the positive class
    precision_norm, recall_norm, pr_thresholds_norm = precision_recall_curve(truth, predict)
    pr_auc_norm = auc(recall_norm, precision_norm)

    # PR curve where "anormal" is the positive class
    precision_anom, recall_anom, pr_thresholds_anom = precision_recall_curve(truth, -predict, pos_label=0)
    pr_auc_anom = auc(recall_anom, precision_anom)

    # print('Current data information:  \t{}-cake{}-{}-{}'.format(run_times, args.num_clusters, dataset_name, flag))
    print('AUROC:{}, AUPR-IN:{}, AUPR-OUT:{}'.format(roc_auc, pr_auc_norm, pr_auc_anom))
    return roc_auc, pr_auc_norm, pr_auc_anom


class LogFormatter:
    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime("%x %X"),
            timedelta(seconds=elapsed_seconds),
        )
        message = record.getMessage()
        message = message.replace("\n", "\n" + " " * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ""


def create_logger(filepath, rank):
    """
    Create a logger.
    Use a different log file for each process.
    """
    # create log formatter
    log_formatter = LogFormatter()

    # create file handler and set level to debug
    if filepath is not None:
        if rank >= 0:
            filepath = "%s-%i" % (filepath, rank)
            file_handler = logging.FileHandler(filepath, "a")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(log_formatter)

    # create console handler and set level to info
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_formatter)

    # create logger and set level to debug
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if filepath is not None:
        logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # reset logger elapsed time
    def reset_time():
        log_formatter.start_time = time.time()

    logger.reset_time = reset_time
    return logger


class PD_Stats(object):
    """
    Log stuff pandas library
    """

    def __init__(self, path, columns):
        self.path = path
        # reload path stats
        if os.path.isfile(self.path):
            self.stats = pd.read_pickle(self.path)

            # check that columns are the same
            assert list(self.stats.columns) == list(columns)
        else:
            self.stats = pd.DataFrame(columns=columns)

    def update(self, row, save=True):
        self.stats.loc[len(self.stats.index)] = row

        # save the statistics
        if save:
            self.stats.to_pickle(self.path)


def initialzie_exp(params, *args, dump_params=False):
    """
    Initialize the experience:
    - dump parameters
    - create checkpoint repo
    - create a logger
    - create a panda object to keep track of the training statistics
    """
    # dump parameters
    if dump_params:
        pickle.dump(params, open(os.path.join(params.dump_path, "params.pkl"), "wb"))

    # create repo to store checkpoints
    params.dump_checkpoints = os.path.join(params.dump_path)
    if not params.rank and not os.path.isdir(params.dump_checkpoints):
        os.mkdir(params.dump_checkpoints)

    # create a panda object to log loss and acc
    training_stats = PD_Stats(
        os.path.join(params.dump_path, "stats" + str(params.rank) + ".pkl"), args
    )
    # create a logger
    logger = create_logger(
        os.path.join(params.dump_path, "train.log"), rank=params.rank

    )
    logger.info("============ Initialized logger ============")
    logger.info(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(params)).items()))
    )
    logger.info("The experiment will be stored in %s\n" % params.dump_path)
    logger.info("")
    return logger, training_stats


class AverageLoss(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# if args.dataset == 'svhn':
#     normalize = transforms.Normalize(mean=[x / 255.0 for x in[109.9, 109.7, 113.8]],
#                                      std=[x / 255.0 for x in [50.1, 50.6, 50.8]])
# else:    cifar10
#     normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
#                                      std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
def cal_affinties(inputs1, inputs2):
    b, d = inputs1.shape
    # affinity_matrix = torch.cdist(embedding_t, embedding_t, p=2).clamp(min=1e-12, max=1e+12)**2
    affinity_matrix = torch.pow(inputs1, 2).sum(dim=1, keepdim=True).expand(b, b) + \
                      torch.pow(inputs2, 2).sum(dim=1, keepdim=True).expand(b, b).t()
    affinity_matrix.addmm_(1, -2, inputs1, inputs2.t())
    return affinity_matrix

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

@torch.no_grad()
def batch_shuffle_ddp(x):
    """
    Batch shuffle, for making use of BatchNorm.
    *** Only support DistributedDataParallel (DDP) model. ***
    """
    # gather from all gpus
    batch_size_this = x.shape[0]
    x_gather = concat_all_gather(x)
    batch_size_all = x_gather.shape[0]

    num_gpus = batch_size_all // batch_size_this

    # random shuffle index
    idx_shuffle = torch.randperm(batch_size_all).cuda()

    # broadcast to all gpus
    torch.distributed.broadcast(idx_shuffle, src=0)

    # index for restoring
    idx_unshuffle = torch.argsort(idx_shuffle)

    # shuffled index for this gpu
    gpu_idx = torch.distributed.get_rank()
    idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

    return x_gather[idx_this], idx_unshuffle

@torch.no_grad()
def batch_unshuffle_ddp(x, idx_unshuffle):
    """
    Undo batch shuffle.
    *** Only support DistributedDataParallel (DDP) model. ***
    """
    # gather from all gpus
    batch_size_this = x.shape[0]
    x_gather = concat_all_gather(x)
    batch_size_all = x_gather.shape[0]

    num_gpus = batch_size_all // batch_size_this

    # restored index for this gpu
    gpu_idx = torch.distributed.get_rank()
    idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

    return x_gather[idx_this]


def get_Mahalanobis_convariance(inputs):
    # embedding_list = []
    # for i, (inputs, _) in enumerate(eval_loader):
    #     model.eval()
    #     with torch.no_grad():
    #         inputs = inputs.cuda(non_blocking=True)
    #         features = model.backbone(inputs)
    #         embeddings = model.projecter(features)
    #         # embeddings = F.avg_pool2d(embeddings, 8).view(-1, 256)
    #         embedding_list.append(embeddings)
    # embeddings = torch.cat(embedding_list, dim=0).detach().cpu().numpy()
    mean_np = np.mean(inputs, axis=0)
    cov_np = np.cov(inputs, rowvar=False)
    m = 1e-9
    cov_np = cov_np + np.eye(cov_np.shape[1]) * m
    inv_cov_np = np.linalg.inv(cov_np)

    return mean_np, inv_cov_np


def mahalanobis(x, u, inverse_covariance):
    delta = x - u
    m = np.dot(np.dot(delta, inverse_covariance), delta)
    return np.sqrt(m)


def evl_ood(train_embedding, test_embeddings):
    train_embedding = F.normalize(train_embedding, dim=1)
    test_embeddings = F.normalize(test_embeddings, dim=1)
    dist = torch.mm(test_embeddings, train_embedding.T)
    score1, _ = torch.max(dist, dim=1)
    score2 = torch.norm(test_embeddings, dim=1)
    score = score1.squeeze()*score2.squeeze()
    return score.numpy()


def adjust_learning_rate_poly(optimizer, epoch, num_epochs, base_lr, power):
    lr = base_lr * (1-epoch/num_epochs)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def patch2image(patchset, weight, imagesize, args):
    rows, cols = imagesize
    w = args.outersize
    r = int(w/2)
    img = np.zeros((rows+2*r, cols+2*r))
    mask = np.zeros((rows+2*r, cols+2*r))
    winmask = np.ones((w, w))
    for i in range(rows):
        for j in range(cols):
            img[i:i+w, j:j+w] = img[i:i+w, j:j+w] + patchset[i*cols+j].reshape(w, w) * weight[i*cols+j]
            mask[i:i+w, j:j+w] = mask[i:i+w, j:j+w] + winmask
    img = img / mask
    img = img[r:r + rows, r:r + cols]
    return img


def plot_loss_curve(loss1, loss2, args, epoch):
    fig = plt.figure(figsize=(7, 5))
    ax1 = fig.add_subplot(1, 1, 1)
    plt.plot(loss1, 'b-', label="Classification Loss")
    plt.plot(loss2, 'r-', label="Recover Loss")
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title('Loss Curve')
    image_path = osp.join(args.dump_path, "{}-{}-loss curve_{}.jpg".format(args.traindata, args.traindata_sub, epoch))
    plt.savefig(image_path)
    plt.close()


