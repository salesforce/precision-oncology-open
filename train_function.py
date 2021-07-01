"""
Function to run one epoch of training for cancer detection using histopathology dataset.

train : function to train one epoch with histopathology data and update tensorboard variables

  Typical usage example:
  Add this function in a epoch loop
    train(model, train_dataset, args, train_loader, epoch, class_weights, optimizer, writer)
"""

from __future__ import print_function

import sys
import os
import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_recall_fscore_support
import torch
from torch.autograd import Variable
import IPython


def train(model, train_dataset, args, train_loader, epoch, class_weights, optimizer, writer):
    """Function to train one epoch with histopathology data and update tensorboard variables
    Args:
        model (torch.nn model) : model for training
        train_dataset (PatchDataset): load patches of test datasets
        args (argparse) : input arguments containing model settings
        train_loader : combines a dataset and a sampler, and provides an iterable over the given dataset.
        epoch (int): epoch number in training
        class_weights (list): the per class weights to balance sampling from each class
        optimizer (torch.optim.optimizer): optimizer that we are going to use for training
        writer (torch.utils.tensorboard.Writer) : TensorBoard writer which will output to ./runs/ directory by default
    """

    model.train()
    train_loss = 0.
    train_error = 0.
    attention_weights_std = 0.
    true_labels, predicted_labels, prediction_probs,  = [], [], []

    sorted_att_files = []
    for batch_idx, (data, label, names) in enumerate(train_loader):
        bag_label = label[0]
        true_labels.append(bag_label.cpu().data.numpy()[0])
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)

        # reset gradients
        optimizer.zero_grad()

        # calculate loss and metrics
        if args.multigpu:
            calculate_objective = model.module.calculate_objective
        else:
            calculate_objective = model.calculate_objective
        loss, error, predicted_label, prediction_prob, attention_weights = \
            calculate_objective(data, bag_label, class_weights)
#       IPython.embed()
        train_loss += loss.data
        predicted_labels.append(predicted_label.cpu().data.numpy())
        prediction_probs.append(prediction_prob.cpu().data.numpy())
        train_error += float(error)
        if args.model == 11:
            attention_weights_std += torch.mean(torch.std(attention_weights, 0)).squeeze().cpu().detach().numpy()
            attention_weights = torch.mean(attention_weights, dim=0)

        attention_weights = attention_weights.squeeze().cpu().detach().numpy()
        attention_weights_scaled = \
            (attention_weights - min(attention_weights))/(max(attention_weights) - min(attention_weights))
        sorted_indices = np.argsort(-attention_weights_scaled)
        threshold_mask = (np.sort(-attention_weights_scaled)*-1.0) > 0.1  # cutoff for patch attention
        sorted_att_files.append(np.array(names)[sorted_indices][threshold_mask].tolist())
        sys.stdout.write("Bags Processed = : %f  \r" % (batch_idx/len(train_loader)))
        sys.stdout.flush()
        # backward pass
        loss.backward()
        # step
        optimizer.step()

    # selecting only the false positive samples
    preds_int = np.array(predicted_labels).reshape(-1).astype(int)
    idx_empty = np.invert((np.array(true_labels) != preds_int) & np.invert(np.array(true_labels).astype(bool)))
    for i in range(len(idx_empty)):
        if idx_empty[i] or (not args.hard_negative):
            sorted_att_files[i] = []

    # calculate loss and error for epoch
#   p, r, f, _ = precision_recall_fscore_support(true_labels, predicted_labels)
    acc = accuracy_score(true_labels, predicted_labels)
#   fpr, tpr, _ = roc_curve(true_labels, prediction_probs)
#   auc_score = auc(fpr, tpr)
#   bacc = (r[0] + r[1])/2.
    train_loss /= len(train_loader)
    train_error /= len(train_loader)
    attention_weights_std /= len(train_loader)
    class_sample_count = np.unique(true_labels, return_counts=True)[1]
    print("Class Sample Count: {}".format(class_sample_count))
    writer.add_scalar('loss_train', train_loss, epoch)
    writer.add_scalar('error_train', train_error, epoch)
#   writer.add_scalar('balanced_accuracy_train', bacc, epoch)
#   writer.add_scalar('auc_train', auc_score, epoch)
    writer.add_scalar('mean_prediction_prob_train', np.mean(prediction_probs), epoch)
    if args.model == 11:
        print('Epoch: {}, Loss: {:.4f}, Train error: {:.4f}, Multihead Std: {:.4f}'
              .format(epoch, train_loss.cpu().numpy()[0], train_error, attention_weights_std))
    else:
        print('Epoch: {}, Loss: {:.4f}, Train error: {:.4f}'.format(epoch, train_loss.cpu().numpy(), train_error))

    train_dataset.hard_mine_samples = sorted_att_files
