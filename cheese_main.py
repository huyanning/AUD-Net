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
# from test import show, scatter_plot
import torch.nn.functional as F
from torchvision import datasets, transforms
from datasets import DualWinHyperLoaderh5, Traindataset
import torch.utils.data as Data
from opt_trainer import opt_sk
from sklearn.metrics import roc_auc_score
from test import anomaly_show, scatter_plot
from sklearn.preprocessing import MinMaxScaler
from model import CHEESE
from utils import patch2image, plot_loss_curve
import matplotlib.pyplot as plt
import scipy.io as sio
import os.path as osp
import torch.cuda.amp as amp
from evaluation import test
from sklearn.metrics import roc_curve, precision_recall_curve, auc



def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
      m.eval()


def generate_dual_window(inputs, targets,  mask):
    mask = mask.cuda()
    innerdata = inputs[:, mask == 1, :]
    outerdata = inputs[:, mask == 0, :]
    innerlabel = targets[:, mask == 1]
    outerlabel = targets[:, mask == 0]
    bs, outernum, feature_dim = outerdata.shape
    _, innernum, _ = innerdata.shape
    outerdata = outerdata.unsqueeze(1).expand(bs, innernum, outernum, feature_dim)
    innerdata = innerdata.unsqueeze(2)
    data = torch.cat([innerdata, outerdata], dim=2).view(-1, outernum+1, feature_dim)
    label = innerlabel.long().view(-1)
    return data, label


def get_score(projections, mask):
    bs, N, dim = projections.shape
    # tmp = mask.sum(dim=0)
    # mask = mask.unsqueeze(0).unsqueeze(2).expand(bs, N, dim)
    # centers = (projections*mask).sum(dim=1, keepdim=True).expand(bs, N, dim) / tmp
    center_index = int(N / 2)
    centers = projections[:, center_index, :].unsqueeze(1).expand(bs, N, dim)
    # projections = F.normalize(projections, dim=2)
    # centers = F.normalize(centers, dim=2)
    logit = (projections * centers).sum(dim=2)
    return 1 - ((1 + logit) / 2)


def get_global_score(projections):
    projections = projections.mean(dim=1)
    N, dim = projections.shape
    center = projections.mean(dim=0, keepdim=True).expand(N, dim)
    # projections = F.normalize(projections, dim=1)
    # center = F.normalize(center, dim=1)
    logit = (projections * center).sum(dim=1)
    return 1 - (1 + logit) / 2


def center_stack(projections, targets, mask, learner, args):
    bs, N, dim = projections.shape
    tmp = mask.sum(dim=0)
    center_index = int(N / 2)
    # center_index = 0
    center_label = targets[:, center_index].unsqueeze(1).expand(bs, N)
    pairlabel = center_label == targets
    pairlabel = pairlabel.float()
    mask = mask > 0
    tmplabel = pairlabel[:, mask]
    anomalyrate = torch.sum(tmplabel, dim=1) / (args.outersize ** 2 - args.innersize ** 2)
    centers = projections[:, center_index, :].unsqueeze(1).expand(bs, N, dim)
    logit = torch.cat((projections, centers), dim=2)
    logit = projections * centers
   # logit = learner.fuse_layer(logit.view(bs*N, -1))
    return logit.view(bs*N, -1), pairlabel, anomalyrate


def evalu(learner, dataloader, count, mask, args):
    print("start evaluation")
    learner.eval()
    feature_list = []
    label_list = []
    prediction_list = []
    score_list = []
    predict_label_list = []
    original_label_list = []
    innermask = mask < 1
    outermask = mask > 0
    for i, (inputs, labels) in enumerate(dataloader):
        bs, neighbor_num, channels, bands = inputs.shape
        with torch.no_grad():
            inputs = inputs.cuda(args.gpu)
            labels = labels.cuda(args.gpu)
            inputs = F.normalize(inputs, dim=1)
            # center_index = int(neighbor_num / 2)
            # centers = inputs[:, center_index, :, :].unsqueeze(1).expand(bs, neighbor_num, channels, bands)
            # centers = centers.unsqueeze(1).expand(bs, neighbor_num, channels, bands)
            centers = inputs[:,innermask, :,:].mean(1)
            centers = centers.unsqueeze(1).expand(bs, neighbor_num, channels, bands)
            inputs = centers*inputs
            # logit = torch.cat([inputs, centers], dim=2)
            # logit = learner.center_stack(inputs, mask)
            feature_maps = learner.encoder(inputs.view(-1, 1, bands))
            features = learner.vladnet(feature_maps)
            logit, pairlabel, anomalyrate = center_stack(features.view(bs, neighbor_num, -1), labels, mask, learner, args)
            # predictions = learner.classifier(features).view(bs, neighbor_num)
            predictions = learner.classifier(logit)
            # predict_label = torch.argmax(predictions, dim=1)
            # predictions = torch.max(predictions, dim=1)[0].view(bs, neighbor_num)
            logit = logit.view(bs, neighbor_num, -1)[:, outermask, :]
            #logit = logit.view(bs, neighbor_num, -1)
            _, anomaly_score = learner.transformer(logit)
            # predictions = learner.classifier(logit).view(bs, neighbor_num)
            # projections = learner.projecter(features).view(bs, neighbor_num, -1)

            # score = 1 - torch.softmax(anomaly_score, dim=1)
            # predictions = get_score(projections, mask)
            # predictions = predictions * score
            label_list.append(pairlabel.view(bs*neighbor_num))
            prediction_list.append(predictions)
            score_list.append(anomaly_score)
            original_label_list.append(labels)
            # predict_label_list.append(predict_label)
            # projections_list.append(logit.cpu())
        if i % 1000 == 0:
            print("Currnt mini-batches: {}".format(i))
    #rows, cols = 940, 475
    rows, cols, bands = dataloader.dataset.datasize
    #rows, cols = 550, 400
    labels = torch.cat(label_list, dim=0).cpu().detach().numpy()
    original_label = torch.cat(original_label_list, dim=0).cpu().detach().numpy()
    tmp = print(len(labels[labels==1]))
    predictions = torch.cat(prediction_list, dim=0).cpu().detach().numpy()
    # predict_label = torch.cat(predict_label_list, dim=0).cpu().detach().numpy()
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # predictions = scaler.fit_transform(predictions.transpose()).transpose()
    score = torch.cat(score_list, dim=0)
    w = args.outersize
    #w = 5
    roc_auc, pr_auc_norm, pr_auc_anom = \
        evaluator(predictions, labels, "featuresum", args)
    label_map = np.zeros((rows*w, cols*w))
    prediction_map = np.zeros((rows *w, cols *w))
    predict_label_map = np.zeros((rows *w, cols *w))
    # accuracy = np.zeros(len(labels))
    # accuracy[predict_label==labels] = 1.
    # accuracy = np.sum(predict_label) / len(predict_label)
    # print("Classification accuracy is: {}".format(accuracy))
    labels = labels.reshape(rows*cols, w*w)
    # predict_label = predict_label.reshape(rows*cols, w*w)
    predictions = predictions.reshape(rows*cols, w*w)

    # roc_auc = np.zeros(rows*cols)
    for i in range(rows):
        for j in range(cols):
            label_map[i*w:(i+1)*w, j*w:(j+1)*w] = labels[cols*i + j, :].reshape(w, w)
            prediction_map[i*w:(i+1)*w, j*w:(j+1)*w] = predictions[cols * i + j, :].reshape(w, w)
            # predict_label_map[i * w:(i + 1) * w, j * w:(j + 1) * w] = predict_label[cols * i + j, :].reshape(w, w)
    #         # fpr, tpr, roc_thresholds = roc_curve(labels[cols*i + j, :], predictions[cols * i + j, :])
    #         # roc_auc[cols*i + j] = auc(fpr, tpr)
    #
    score_map = score.reshape(rows, cols).cpu().detach().numpy()
    # print(roc_auc.mean(axis=0))
    # predict_map = predictions.reshape(940, 475).cpu().detach().numpy()
    # score_map = score.reshape(550, 400).cpu().detach().numpy()
    # predict_map = predictions.reshape(550, 400).cpu().detach().numpy()
    # print("Evaluation accuracy: {}".format(torch.sum((predictions.squeeze()-labels.squeeze())**2)))
    # predict_labels = torch.argmax(predictions, dim=1)
    # accuracy = torch.zeros(len(labels))
    # accuracy[labels == predict_labels] = 1.
    # accuracy = torch.sum(accuracy, dim=0) / len(labels)
    # print("Classification accuracy is: {}".format(accuracy))
    # predict_labels = predictions.view(-1, args.outersize**2).sum(dim=1)
    # labels = labels.view(-1, args.outersize**2).sum(dim=1)
    # labels = labels.cpu().detach().numpy()
    # predict_labels = predict_labels.cpu().detach().numpy()
    # groundtruth = labels.reshape(940, 475)
    # predict_maps = predict_labels.reshape(940, 475)
    # predictions =predictions.reshape(rows*cols*w*w)
    # labels = labels.reshape(rows*cols*w*w)
    plt.figure(1)
    plt.subplot(1, 4, 1)
    plt.imshow(label_map)
    plt.subplot(1, 4, 2)
    plt.imshow(prediction_map)
    plt.subplot(1, 4, 3)
    plt.imshow(score_map)
    plt.subplot(1, 4, 4)
    plt.imshow(predict_label_map)
    image_path = osp.join(args.dump_path, "evalution_{}.jpg".format(count))
    plt.savefig(image_path, dpi=800)
    plt.close()
    zero_mask = original_label>0
    zero_mask = zero_mask.flatten()
    new_labels=labels.flatten()[zero_mask]
    new_predictions = predictions.flatten()[zero_mask]
    roc_auc, pr_auc_norm, pr_auc_anom = \
        evaluator(new_predictions, new_labels, "featuresum", args)
    sio.savemat(osp.join(args.dump_path, "predictions_{}_honghu_longkou.mat".format(count)), {'predictions': new_predictions})
    sio.savemat(osp.join(args.dump_path, "labels_{}_honghu_longkou.mat".format(count)), {'labels': new_labels})
    #roc_auc, pr_auc_norm, pr_auc_anom = \
    #    evaluator(predictions, labels, "featuresum", args)
    # roc_auc, pr_auc_norm, pr_auc_anom = \
    #     evaluator(score.cpu().detach().numpy(), anomalyrate.cpu().detach().numpy(), "featuresum", args)


def detector(learner, dataloader, count, mask, args):
    print("start detecting...")
    learner.eval()
    feature_list = []
    label_list = []
    prediction_list = []
    score_list = []
    projections_list = []
    for i, (inputs, labels) in enumerate(dataloader):
        bs, neighbor_num, channels, bands = inputs.shape
        with torch.no_grad():
            inputs = inputs.cuda(args.gpu)
            labels = labels.cuda(args.gpu)
            feature_maps = learner.encoder(inputs.view(-1, 1, bands))
            features = learner.vladnet(feature_maps)
            projections = learner.projecter(features).view(bs, neighbor_num, -1)
            predictions = get_score(projections, mask)
            _, anomaly_score = learner.transformer(features.view(bs, neighbor_num, -1))
            score_list.append(anomaly_score)
            label_list.append(labels)
            prediction_list.append(predictions)
            projections_list.append(projections)
        if i % 1000 == 0:
            print("Currnt mini-batches: {}".format(i))
    rows, cols = args.rows, args.cols
    predictions = torch.cat(prediction_list, dim=0)
    projections = torch.cat(projections_list, dim=0)
    global_score = get_global_score(projections)
    global_score = (global_score - global_score.mean(dim=0))**2
    # combine_predictions = -(predictions - global_score.unsqueeze(1).expand_as(predictions))

    # predictions, _ = torch.max(predictions, dim=1)
    # predictions = predictions.view(rows * cols, args.outersize**2)
    # predictions = torch.sum(predictions, dim=1)/predictions.shape[1]
    labels = torch.cat(label_list, dim=0)
    score = torch.cat(score_list, dim=0)
    labels = labels.cpu().detach().numpy()
    score = score.cpu().detach().numpy()
    predictions = predictions.cpu().detach().numpy()
    global_score = global_score.cpu().detach().numpy()
    # combine_predictions = combine_predictions.cpu().detach().numpy()
    img = patch2image(predictions, score, [rows, cols], args)
    # predictions = np.sum(predictions, axis=1)
    predictions = np.sqrt(np.sum(predictions*predictions, axis=1))
    weighted_predictions = img.reshape(rows * cols)
    combine_predictions = predictions + global_score
    img_combine = patch2image(combine_predictions, score, [rows, cols], args)
    # combine_predictions = np.sum(combine_predictions, axis=1) / args.outersize ** 2
    combine_weighted_predictions = img_combine.reshape(rows * cols)
    # combine_weighted_predictions = combine_predictions * score.squeeze()
    # for i in range(len(test_features)):
    #     m_distance[i] = mahalanobis(test_features[i], mean_train, inv_cov_train)
    # features = torch.cat(feature_list, dim=0).cpu().detach().numpy()
    # labels = torch.cat(label_list, dim=1).cpu().detach().numpy()
    # scatter_plot(encodes1, label, epoch, args)
    # scatter_show(y_train, targets, encodes2, 2, epoch)
    # histogram_modified(errors[y_train == 1], errors[y_train==0], args, epoch)
    # show(targets, embeddings1, epoch, args)

    roc_auc, pr_auc_norm, pr_auc_anom = \
        evaluator(score, labels, "score", args)
    roc_auc_p, pr_auc_norm_p, pr_auc_anom_p = \
        evaluator(predictions, labels, "predictions", args)
    roc_auc_w, pr_auc_norm_w, pr_auc_anom_w = \
        evaluator(weighted_predictions, labels, "weighted_predictions", args)
    roc_auc_g, pr_auc_norm_g, pr_auc_anom_g = \
        evaluator(global_score, labels, "global_score", args)
    roc_auc_c, pr_auc_norm_c, pr_auc_anom_c = \
        evaluator(combine_predictions, labels, "combine_predictions", args)
    roc_auc_cw, pr_auc_norm_cw, pr_auc_anom_cw = \
        evaluator(combine_weighted_predictions, labels, "combine_weighted_predictions", args)
    anomaly_show(predictions, predictions, weighted_predictions, global_score, combine_predictions,
                 predictions,
                 count, args)


def cheese_experiment(run_times, args):
    args.run_times = run_times
    # print(torch.cuda.is_available())
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    running_info = '{}-cheese{}-{}-{}'.format(run_times, args.num_clusters, args.traindata, args.traindata_sub)
    model_path = os.path.join(args.dump_path, running_info)
    # train_data = Traindataset(args, args.traindata, args.traindata_sub)

    train_data = DualWinHyperLoaderh5(args, train=True, stride=3)
    # plt.figure(1)
    # plt.subplot(2, 3, 1)
    # plt.imshow(train_data.anomalyrate.reshape(550, 400), label='read_errors')
    # plt.show()
    test_data = DualWinHyperLoaderh5(args, train=False, stride=3)
    #test_data = Traindataset(args, args.testdata, args.testdata_sub)
    # test_data = train_data
    # train_data.data = train_data.data[:int(len(train_data.data)/2)]
    ###########################



    ############################
    print("Training data size: {}".format(train_data.datasize))
    print("Testing data size: {}".format(test_data.datasize))
    args.num_neighbors = args.outersize**2 - args.innersize**2
    args.rows, args.cols, args.bands = test_data.datasize
    train_loader = Data.DataLoader(dataset=train_data,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   pin_memory=True,
                                   num_workers=args.num_work)
    # eval_loader = Data.DataLoader(dataset=evaldata,
    #                                batch_size=args.batch_size,
    #                                shuffle=False,
    #                                pin_memory=True,
    #                                num_workers=args.num_work)
    test_loader = Data.DataLoader(dataset=test_data,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=args.num_work)


    learner = CHEESE(args).cuda(args.gpu)
    # learner = torch.nn.DataParallel(learner, device_ids=range(torch.cuda.device_count())).cuda()
    # summary(E, (1, 32, 32))
    optimizer = optim.Adam(learner.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)
    # optimizer = optim.SGD(learner.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    load_checkpoint(learner, optimizer, model_path, args, 0)
    start_epoch = args.start_epoch
    start_episode = args.start_episode
    auroc_buf = []
    prin_buf = []
    prout_buf = []
    # train
    print("train starting.........")
    # learner.apply(set_bn_eval)
    # for episode in range(args.start_episode, args.EPISODE):
    loss1_list = []
    loss2_list = []
    scaler = amp.GradScaler()
    mask = train_data.mask.cuda(args.gpu)
    # test(8, args)
    # evalu(learner, test_loader, 19, mask, args)
    for epoch in range(start_epoch, args.epochs):
        learner.train()
        cross_entropy_loss = AverageLoss()
        recover_loss = AverageLoss()
        contrastive_loss = AverageLoss()
        batch_time = AverageLoss()
        end = time.time()
        for i, (inputs, labels) in enumerate(train_loader):

            # inputs, labels = train_loader.__iter__().next()
            inputs = inputs.cuda(args.gpu)
            labels = labels.cuda(args.gpu)
            with amp.autocast():
                loss, loss_list = learner(inputs, labels, mask)
            loss1 = loss_list[0]
            loss2 = loss_list[1]
            loss3 = loss_list[2]
            # loss1_list.append(loss1)
            # loss2_list.append(loss2)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # optimizer.step()
            cross_entropy_loss.update(loss1)
            recover_loss.update(loss2)
            contrastive_loss.update(loss3)
            # triplet_loss.update(loss_dic[1])
            batch_time.update(time.time() - end)
            end = time.time()
            adjust_learning_rate_poly(optimizer, i+len(train_loader)*epoch, args.epochs*len(train_loader), args.lr, 2)
        # adjust_learning_rate(optimizer, episode, args)

        #features, clusters, indices, _, _ = get_features(learner, train_loader)

            if i % 100 == 99:
                print('Current data information:  \t{}'.format(running_info))
                print('Epoch: [{}/{}] Episode: [{}/{}]\t'
                      'CrossEntropy Loss {:.5f}\t'
                      'Recover Loss {:.5f}\t'
                      'Contrastive Loss {:.5f}\t'
                      .format(epoch, args.epochs, i + 1, len(train_loader), cross_entropy_loss.avg, recover_loss.avg,
                              contrastive_loss.avg))

        save_checkpoint({
            'current_running_times': run_times,
            # 'dataset_name': args.dataset,
            'epoch': epoch+1,
            'episode': i,
            'state_dict': learner.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best=False, epoch=epoch, filepath=model_path)
        # evalu(learner, test_loader, epoch, mask, args)
        test(epoch, args)
        # detector(learner, test_loader, i+len(test_loader)*epoch, mask, args)
        # plot_loss_curve(loss1_list, loss2_list, args, epoch)







