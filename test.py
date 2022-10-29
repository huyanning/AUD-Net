# coding = coding=utf-8
import torch
import os
import numpy as np
from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
# matplotlib.use('Agg')
from matplotlib.pyplot import plot, savefig, cla
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os.path as osp
from sklearn import metrics
from sklearn.decomposition import PCA
import torch.utils.data as Data
import scipy.io as sio


def detector(test_loader, net, log_dir, groundtruth, epoch, arg):
    print("Starting detecting......")
    G, D, C = net
    latent_size = 32
    rows, cols = groundtruth.shape
    label = groundtruth.reshape(rows * cols)


    if os.path.exists(log_dir):
        checkpoint = torch.load(log_dir)
        G.load_state_dict(checkpoint['model_state_dict_G'])
        D.load_state_dict(checkpoint['model_state_dict_D'])
        C.load_state_dict(checkpoint['model_state_dict_C'])
        print("Successful loading Detector!")
    else:
        print("no exit detector in this directory")
        exit()

    output_response = []
    true_sample = []
    fake_sample = []
    label1_tuple = []
    predict_real_tuple = []
    predict_fake_tuple = []
    cos = nn.CosineSimilarity().cuda()
    for i, (data, _) in enumerate(test_loader):
        z = torch.randn((data.shape[0], latent_size)).cuda()
        G_sample = G(z)
        data = data.cuda()
        feature_real, predict_real = D(data)
        feature_fake, predict_fake = D(G_sample)
        decode_real = C(data.squeeze())
        # re_error = cos(decode_real.squeeze(), data.squeeze())
        # re_error = F.mse_loss(decode_real.squeeze(), data.squeeze())
        # divergence = torch.mean(0.5 * torch.sum(torch.exp(C.log_sigma) + torch.pow(C.mu, 2) - 1. - C.log_sigma, dim=1), dim=0)
        att1 = torch.norm(decode_real.squeeze() - data.squeeze(), dim=1)
        output_response.append(predict_real.cpu().detach().numpy())
        fake_sample.append(G_sample.squeeze().cpu().detach().numpy())
        true_sample.append(data.squeeze().cpu().detach().numpy())
        label1_tuple.append(att1.squeeze().cpu().detach().numpy())
        predict_real_tuple.append(predict_real.squeeze().cpu().detach().numpy())
        predict_fake_tuple.append(predict_fake.squeeze().cpu().detach().numpy())

    output_response = np.vstack(output_response)
    true_sample = np.vstack(true_sample)
    fake_sample = np.vstack(fake_sample)
    label1 = np.hstack(label1_tuple)
    predict_real = np.hstack(predict_real_tuple)
    predict_fake = np.hstack(predict_fake_tuple)
    # label1 = MinMaxScaler(feature_range=(0, 1)).fit_transform(label1.reshape(-1, 1))

    # response map visualization
    img1 = output_response.reshape(rows, cols).transpose()
    img2 = label1.reshape(rows, cols).transpose()
    img_f = img1 + img2
    plt.figure(1)
    plt.cla()
    plt.subplot(2, 2, 1)
    plt.imshow(img1)
    plt.xlabel('D_predcit')
    plt.subplot(2, 2, 3)
    plt.imshow(img_f)
    plt.xlabel('Fuse response')
    plt.subplot(2, 2, 4)
    plt.imshow(img2)
    plt.xlabel('likelihood')
    image_path = osp.join(arg.dir, "response_{0}.jpg".format(epoch))
    plt.savefig(image_path, dpi=800)
    plt.close()

#    # TSNE map visualization
#    index = np.random.randint(rows * cols, size=5000)
#    x_y = np.vstack((encode_real[index], encode_fake[index]))
#    x_y_TSNE = TSNE(n_components=2).fit_transform(x_y)
##    # x_y_TSNE = MinMaxScaler(feature_range=(-1, 1)).fit_transform(x_y_TSNE)
#    x_TSNE = x_y_TSNE[:5000]
#    y_TSNE = x_y_TSNE[5000:]
#    plt.figure(2)
#    plt.cla()
#    plt.scatter(x_TSNE[:, 0], x_TSNE[:, 1], c='green', label='Real_sample', marker='.', edgecolors='none', alpha=0.2)
#    plt.scatter(y_TSNE[:, 0], y_TSNE[:, 1], c='red', label='Fake_sample', marker='.', edgecolors='none', alpha=0.2)
#    plt.legend(fontsize='xx-small', loc=4)
#    image_path = osp.join("log", "scatter_encode_{0}.jpg".format(epoch))
#    plt.savefig(image_path, dpi=800)
#    plt.close()
#
#    # index = np.random.randint(rows * cols, size=10000)
#    x_y = np.vstack((true_sample[index], fake_sample[index]))
#    x_y_TSNE = TSNE(n_components=2).fit_transform(x_y)
##    # x_y_TSNE = MinMaxScaler(feature_range=(-1, 1)).fit_transform(x_y_TSNE)
#    x_TSNE = x_y_TSNE[:5000]
#    y_TSNE = x_y_TSNE[5000:]
#    plt.figure(3)
#    plt.cla()
#    plt.scatter(x_TSNE[:, 0], x_TSNE[:, 1], c='green', label='Real_sample', marker='.', edgecolors='none', alpha=0.2)
#    plt.scatter(y_TSNE[:, 0], y_TSNE[:, 1], c='red', label='Fake_sample', marker='.', edgecolors='none', alpha=0.2)
#    plt.legend(fontsize='xx-small', loc=4)
#    image_path = osp.join("log", "scatter_original_{0}.jpg".format(epoch))
#    plt.savefig(image_path, dpi=800)
#    plt.close()

    # ROC curves visualization
    fpr1, tpr1, thresholds1 = metrics.roc_curve(label, img1.reshape(rows * cols))
    # fpr2, tpr2, thresholds2 = metrics.roc_curve(label, img_c.reshape(rows * cols))
    fpr3, tpr3, thresholds3 = metrics.roc_curve(label, img2.reshape(rows * cols))
    fpr4, tpr4, thresholds4 = metrics.roc_curve(label, img_f.reshape(rows * cols))

    auc1 = metrics.auc(fpr1, tpr1)
    # auc2 = metrics.auc(fpr2, tpr2)
    auc3 = metrics.auc(fpr3, tpr3)
    auc4 = metrics.auc(fpr4, tpr4)

    index1 = np.argmin(np.abs(tpr1 - 0.95))
    # index2 = np.argmin(np.abs(tpr2 - 0.95))
    index3 = np.argmin(np.abs(tpr3 - 0.95))
    index4 = np.argmin(np.abs(tpr4 - 0.95))

    fpr95_1 = fpr1[index1]
    # fpr95_2 = fpr2[index2]
    fpr95_3 = fpr3[index3]
    fpr95_4 = fpr4[index4]

    plt.figure(4)
    plt.cla()
    plt.plot(fpr1, tpr1, c='r', lw=2, alpha=0.7, label='AUC  =%.4f' % auc1)
    # plt.plot(fpr2, tpr2, c='b', lw=2, alpha=0.7, label='AUC  =%.4f' % auc2)
    plt.plot(fpr3, tpr3, c='g', lw=2, alpha=0.7, label='AUC  =%.4f' % auc3)
    plt.plot(fpr4, tpr4, c='y', lw=2, alpha=0.7, label='AUC  =%.4f' % auc4)
    
    plt.plot(fpr1, tpr1, c='r', lw=2, alpha=0.7, label='FPR95=%.4f' % fpr95_1)
    # plt.plot(fpr2, tpr2, c='b', lw=2, alpha=0.7, label='FPR95=%.4f' % fpr95_2)
    plt.plot(fpr3, tpr3, c='g', lw=2, alpha=0.7, label='FPR95=%.4f' % fpr95_3)
    plt.plot(fpr4, tpr4, c='y', lw=2, alpha=0.7, label='FPR95=%.4f' % fpr95_4)
    
    plt.plot((0, 1), (0, 1), c='#808080', lw=1, ls='--', alpha=0.7)
    plt.xlim((-0.01, 1.02))
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('False Positive Rate', fontsize=13)
    plt.ylabel('True Positive Rate', fontsize=13)
    plt.grid(b=True, ls=':')
    plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
    plt.title('Sandiego ROC', fontsize=17)
    image_path = osp.join(arg.dir, "ROC_{0}.jpg".format(epoch))
    plt.savefig(image_path, dpi=800)
    plt.close()

    plt.figure(5)
    # plt.cla()
    plt.hist(predict_real, bins=100, histtype='bar', label='True Sample', alpha=0.5)
    plt.hist(predict_fake, bins=100, histtype='bar', label='Fake Sample', alpha=0.5)
    plt.legend(fontsize='xx-small', loc='upper right')
    image_path = osp.join(arg.dir, "hist_predict_posneg_{0}.jpg".format(epoch))
    plt.savefig(image_path)
    plt.close()

#    plt.figure(6)
#    index = np.random.randint(rows * cols, size=10000)
#    x = PCA(n_components=2).fit_transform(fake_sample[index])
#    y = PCA(n_components=2).fit_transform(true_sample[index])
#    plt.subplot(1, 2, 1)
#    plt.scatter(x[np.argwhere(label2 > 0.8), 0], x[np.argwhere(label2 > 0.8), 1],
#                s=10, c='blue', label='[0.8, 1]', marker='.', alpha=0.9, edgecolors='none')
#    plt.scatter(x[np.argwhere(label2 < 0.3), 0], x[np.argwhere(label2 < 0.3), 1],
#                s=10, c='red', label='[0, 0.2]', marker='.', alpha=0.9, edgecolors='none')
#    plt.scatter(x[np.argwhere((label2 > 0.3) & (label2 < 0.8)), 0],
#                x[np.argwhere((label2 > 0.3) & (label2 < 0.8)), 1],
#                s=10, c='green', label='[0.2, 0.8]', marker='.', alpha=0.9, edgecolors='none')
#
#    plt.legend(fontsize='xx-small', loc=4)
#    plt.subplot(1, 2, 2)
#    plt.scatter(y[np.argwhere(label1 > 0.8), 0], y[np.argwhere(label1 > 0.8), 1],
#                s=10, c='blue', label='[0.8, 1]', marker='.', alpha=0.9, edgecolors='none')
#    plt.scatter(y[np.argwhere(label1 < 0.3), 0], y[np.argwhere(label1 < 0.3), 1],
#                s=10, c='red', label='[0, 0.2]', marker='.', alpha=0.9, edgecolors='none')
#    plt.scatter(y[np.argwhere((label1 > 0.3) & (label1 < 0.8)), 0],
#                y[np.argwhere((label1 > 0.3) & (label1 < 0.8)), 1],
#                s=10, c='green', label='[0.2, 0.8]', marker='.', alpha=0.9, edgecolors='none')
#
#    plt.legend(fontsize='xx-small', loc=4)
#    image_path = osp.join("log", "scatter_error_{0}.jpg".format(epoch))
#    plt.savefig(image_path, dpi=800)
#    plt.close()
#    return output_response


def anomaly_show(score, predictions, weighted_predictions, global_score, combine_predictions,
                 combine_weighted_predictions, epoch, args):
    # decode_anomaly = np.vstack(decode_anomaly_tuple)
    print("Starting detecting......")
    rows, cols, bands = args.rows, args.cols, args.bands
    # groundtruth = labels.reshape(rows, cols)
    score_map = score.reshape(rows, cols)
    wp_map = weighted_predictions.reshape(rows, cols)
    predictions_map = predictions.reshape(rows, cols)
    global_score_map = global_score.reshape(rows, cols)
    combine_predictions_map = combine_predictions.reshape(rows, cols)
    combine_weighted_predictions_map = combine_weighted_predictions.reshape(rows, cols)

    # sio.savemat(osp.join(args.dump_path, "mahalanobis_{}.mat".format(epoch)), {'mahalanobis': mah_distance})
    sio.savemat(osp.join(args.dump_path, "score_{}.mat".format(epoch)), {'score': score})
    # sio.savemat(osp.join(args.dump_path, "labels.mat"), {'labels': labels})
    sio.savemat(osp.join(args.dump_path, "predictions_{}.mat".format(epoch)), {'predictions': predictions})
    sio.savemat(osp.join(args.dump_path, "weighted_predictions_{}.mat".format(epoch)),
                {'weighted_predictions': weighted_predictions})

    plt.figure(1)
    plt.subplot(2, 3, 1)
    plt.imshow(score_map, label='score')
    plt.xlabel('score')
    # plt.subplot(1, 3, 2)
    # plt.imshow(groundtruth, label='groundtruth')
    # plt.xlabel('groundtruth')
    # plt.subplot(2, 3, 5)
    # plt.imshow(mah_distance_map.transpose(), label='mahalanobis')
    # plt.xlabel('mahalanobis')
    plt.subplot(2, 3, 2)
    plt.imshow(predictions_map, label='predictions')
    plt.xlabel('predictions')
    plt.subplot(2, 3, 3)
    plt.imshow(wp_map, label='weighted predictions')
    plt.xlabel('weighted predictions')
    plt.subplot(2, 3, 4)
    plt.imshow(global_score_map, label='global_score')
    plt.xlabel('global_score')
    plt.subplot(2, 3, 5)
    plt.imshow(combine_predictions_map, label='combine_predictions')
    plt.xlabel('combine_predictions')
    plt.subplot(2, 3, 6)
    plt.imshow(combine_weighted_predictions_map, label='combine_weighted_predictions')
    plt.xlabel('combine_weighted_predictions')
    image_path = osp.join(args.dump_path, "{}-{}-anomaly curve_{}.jpg".format(args.testdata, args.testdata_sub, epoch))
    plt.savefig(image_path, dpi=800)
    plt.close()
    # inputs_background_mean = np.mean(inputs[labels!=1].squeeze(), axis=0)
    # inputs_anomaly_mean = np.mean(inputs[labels==1].squeeze(), axis=0)
    # decodes_background_mean = np.mean(decodes[labels!=1].squeeze(), axis=0)
    # decodes_anomaly_mean = np.mean(decodes[labels==1].squeeze(), axis=0)
    # plt.figure(2)
    # plt.subplot(1, 3, 1)
    # plt.plot(inputs_background_mean.transpose(), color='blue', label='background')
    # plt.plot(inputs_anomaly_mean.transpose(), color='red', label='anomaly')
    # plt.xlabel('inputs')
    # plt.subplot(1, 3, 2)
    # plt.plot(decodes_background_mean.transpose(), color='blue', label='background')
    # plt.plot(decodes_anomaly_mean.transpose(), color='red', label='anomaly')
    # plt.xlabel('decodes')
    # # plt.subplot(1, 3, 3)
    # # plt.plot(memory.T)
    # # plt.plot(decodes_anomaly_mean.transpose(), color='red', label='anomaly')
    # # plt.xlabel('decodes')
    # image_path = osp.join(args.dump_path, "plot curve_{}.jpg".format(epoch))
    # plt.savefig(image_path, dpi=800)
    # plt.close()




def scatter_plot(inputs, label, epoch, args):
    inputs = TSNE(n_components=2, random_state=0).fit_transform(inputs)
    # tmp = np.linalg.norm(input, axis=1)
    # input = input / np.tile(tmp, (3, 1)).transpose()
    input_posi = inputs[label == 1]
    input_nega = inputs[label == 0]

    fig = plt.figure(1, figsize=(4, 4))
    ax = plt.subplot(aspect='equal')
    # ax = fig.add_subplot(projection='3d')
    ax.scatter(input_posi[:, 0], input_posi[:, 1], marker='.', s=2,
                color='red', alpha=0.5, label='x1 samples')
    ax.scatter(input_nega[:, 0], input_nega[:, 1],  marker='.', s=2,
                color='blue', alpha=0.5, label='x2 samples')
    # plt.scatter(x3_samples[:, 0], x1_samples[:, 1], marker='^',
    #             color='red', alpha=0.7, label='x3 samples')
    plt.axis('off')
    # plt.title('Basic scatter plot')
    # plt.ylabel('variable X')
    # plt.xlabel('Variable Y')
    # plt.legend(fontsize='xx-small', loc=4)
    image_path = osp.join(args.dump_path, "feature_scatter_{}.png"
                          .format(epoch))
    plt.savefig(image_path, dpi=800)
    plt.close()

