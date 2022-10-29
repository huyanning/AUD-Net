# coding=utf-8
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import utils as hyper
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.manifold import TSNE
from ROC_AUC import ROC_AUC


def result_show(oput_respons, oput_label, groundtruth, train_num):

    rows, cols = groundtruth.shape
    bg_dic_show = np.zeros((1, rows * cols))
    tg_dic_show = np.zeros((1, rows * cols))
    bg_dic_show = bg_dic_show.reshape(rows, cols)
    tg_dic_show = tg_dic_show.reshape(rows, cols)
    train_show = np.zeros((rows, cols))
    train_show[0:int(train_num/cols), 0:cols] = groundtruth[0:int(train_num/cols), 0:cols]

    plt.figure(1)
    plt.subplot(2, 2, 1)
    plt.imshow(train_show)
    plt.xlabel('Training samples')
    plt.subplot(2, 2, 2)
    plt.imshow(groundtruth)
    plt.xlabel('Groundtruth')
    plt.subplot(2, 2, 3)
    plt.imshow(oput_respons.reshape(rows, cols))
    plt.xlabel('Output response')
    plt.subplot(2, 2, 4)
    plt.imshow(oput_label.reshape(rows, cols))
    plt.xlabel('Output label')

    plt.show()


def datashow(data3d):
    rows, cols, bands = data3d.shape
    num = int(round(np.sqrt(bands)))
    img = np.zeros((int(num*rows), int(num*cols)))
    for i in range(num):
        for j in range(num):
            if i*num+j >= bands:
                break
            else:
                tmp = hyper.convetimage(data3d[:, :, i*num+j])
                # tmp = data3d[:, :, i*num+j]
                img[(i*rows):(i+1)*rows, (j*cols):(j+1)*cols] = tmp

    return img
    # plt.figure(1)
    # plt.imshow(img)
    # plt.xlabel("bands show")


# def TSNE_show(data, label):
#     X = TSNE(n_components=3, random_state=15).fit_transform(data)
#     font = {"color": "darkred",
#             "size": "13",
#             "family": "serif"}
#     # plt.style.use("dark_background")
#     fig = plt.figure(figsize=(8.5, 4))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=label, marker='o')
#     # plt.scatter(X[:, 0], X[:, 1], c=label, alpha=0.6, cmap=plt.cm.get_cmap('rainbow', 10))
#     plt.title("t-SNE", fontdict=font)
#     # cbar = plt.colorbar(ticks=range(10))
#     # cbar.set_label(label="digit value", fontdict=font)
#     # plt.clim(-0.5, 9.5)
#     plt.tight_layout()
#     plt.show()

def show_everything(data3d, groundtruth):
    rows, cols, bands = data3d.shape
    label = groundtruth.reshape(rows*cols)
    output_dic = sio.loadmat("output.mat")
    response = np.array(output_dic['pretrain_response'])
    img_pos = np.array(output_dic['pos_sam'])
    img_nega = np.array(output_dic['neg_sam'])
    data2d = np.array(output_dic['data2d'])
    pretest_encode = np.array(output_dic['pretest_encode'])
    dec_encode = np.array(output_dic['detect_encode'])
    dec_result = np.array(output_dic['detect_result'])
    dec_label = np.array(output_dic['dec_label'])
    predec_label = np.array(output_dic['predec_label'])
    # predec_label = predec_label + 1
    print(predec_label)

    # plt.style.use('fivethirtyeight')
    plt.figure(1)
    fig, ax = plt.subplots(nrows=3, ncols=3)
    ax[0, 0].imshow(data3d[:, :, 20])
    ax[0, 0].set_xlabel('Original Samples band-56')
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    ax[0, 1].imshow(img_pos.reshape(rows, cols).transpose())
    ax[0, 1].set_xlabel('Positive Samples')
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])
    ax[0, 2].imshow(img_nega.reshape(rows, cols).transpose())
    ax[0, 2].set_xlabel('Negative Samples')
    ax[0, 2].set_xticks([])
    ax[0, 2].set_yticks([])
    ax[1, 0].imshow(groundtruth)
    ax[1, 0].set_xlabel('Ground Truth')
    ax[1, 0].set_xticks([])
    ax[1, 0].set_yticks([])
    ax[1, 1].imshow(dec_label.reshape(rows, cols).transpose())
    ax[1, 1].set_xlabel('Label Results')
    ax[1, 1].set_xticks([])
    ax[1, 1].set_yticks([])
    ax[1, 2].imshow(predec_label.reshape(rows, cols).transpose())
    ax[1, 2].set_xlabel('Pre-detected Label')
    ax[1, 2].set_xticks([])
    ax[1, 2].set_yticks([])
    ax[2, 0].imshow(response.reshape(rows, cols).transpose())
    ax[2, 0].set_xlabel('Reconstructed image error')
    ax[2, 0].set_xticks([])
    ax[2, 0].set_yticks([])
    ax[2, 1].imshow(dec_result.reshape(rows, cols).transpose())
    ax[2, 1].set_xlabel('Response image')
    ax[2, 1].set_xticks([])
    ax[2, 1].set_yticks([])
    ax[2, 2].imshow(np.zeros((rows, cols)))
    # ax[2, 2].set_xlabel('Response image')
    ax[2, 2].set_xticks([])
    ax[2, 2].set_yticks([])
    plt.show()
    plt.figure(2)
    print(ROC_AUC(dec_result.reshape(rows, cols).transpose().reshape(rows*cols),
                  response.reshape(rows, cols).transpose().reshape(rows*cols),
                  groundtruth, 2))
    plt.figure(3)
    #
    X = TSNE(n_components=2, random_state=5).fit_transform(data2d)
    Y = TSNE(n_components=2, random_state=5).fit_transform(pretest_encode)
    Z = TSNE(n_components=2, random_state=5).fit_transform(dec_encode)
    # font = {"color": "darkred",
    #         "size": "13",
    #         "family": "serif"}
    # # plt.style.use("dark_background")
    # plt.subplot(1, 3, 1)
    # plt.scatter(X[:, 0], X[:, 1], c=label, marker='.')
    # plt.xlabel("Original distribution")
    # plt.subplot(1, 3, 2)
    # plt.scatter(Y[:, 0], Y[:, 1], c=label, marker='.')
    # plt.xlabel("Distribution in Embedding")
    # plt.subplot(1, 3, 3)
    # plt.scatter(Z[:, 0], Z[:, 1], c=label, marker='.')
    # plt.xlabel("Distribution in Embedding with triplet loss")


    # plt.title("t-SNE", fontdict=font)
    # plt.subplot(1, 3, 2)
    # ax2.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=label.numpy(), marker='o')
    # plt.title("t-SNE", fontdict=font)
    # plt.subplot(1, 3, 3)
    # ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], c=label.numpy(), marker='o')
    # plt.title("t-SNE", fontdict=font)
    # plt.scatter(X[:, 0], X[:, 1], c=label, alpha=0.6, cmap=plt.cm.get_cmap('rainbow', 10))

    # cbar = plt.colorbar(ticks=range(10))
    # cbar.set_label(label="digit value", fontdict=font)
    # plt.clim(-0.5, 9.5)
    plt.tight_layout()
    plt.show()
