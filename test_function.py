"""
Function to run one epoch of testing for cancer detection using histopathology dataset.

metrics : function to calculate metrics used for testing
test : function to test one epoch with histopathology data and update tensorboard variables

  Typical usage example:

  acc, test_error, auc_score, confusion = metrics(prediction_probs, true_labels)
  checkpoint_variables_dict = test(model, args, epoch, test_loader, test_dataset, optimizer, writer, checkpoint_variables_dict)
"""

from __future__ import print_function

import sys
import os
import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, precision_recall_fscore_support
import torch
from sklearn.metrics import classification_report
from torch.autograd import Variable
import IPython


def metrics_binary(prediction_probs, true_labels):
    """Function to calculate metrics for testing, in a binary case.
    Args:
        prediction_probs (torch.nn model) : per class probability
        true_labels (argparse) : input arguments containing model settings
    Returns:
        acc (float) : accuracy
        test_error (float) : test error
        auc_score (float) : AUC (Area under curve) score
        confusion (list[list]) : confusion matrix
    """

    predicted_labels = np.zeros(len(prediction_probs))
    super_threshold_indices = prediction_probs > 0.5
    predicted_labels[super_threshold_indices] = 1
    p, r, f, _ = precision_recall_fscore_support(true_labels, predicted_labels)
    print('\np, r, f', p, r, f)
    acc = accuracy_score(true_labels, predicted_labels)
    test_error = 1.0 - acc
    fpr, tpr, _ = roc_curve(true_labels, prediction_probs)
    auc_score = auc(fpr, tpr)
    bacc = 0.0  # (r[0]+r[1])/2.
    confusion = confusion_matrix(true_labels, predicted_labels)
    return acc, test_error, auc_score, confusion


def metrics(prediction_probs, true_labels):
    """Function 
    Args:
        prediction_probs (torch.nn model) : per class probability
        true_labels (argparse) : input arguments containing model settings
    Returns:
        acc (float) : accuracy
        test_error (float) : test error
    """
    predicted_labels = np.argmax(prediction_probs, axis=1)
    acc = accuracy_score(true_labels, predicted_labels)
    test_error = 1.0 - acc
    target_names = ['primary gleason 3', 'primary gleason 4 + 5']
    #target_names = ['2', '3', '4', '5', '6', '7', '8', '9', '10']
    print(classification_report(true_labels, predicted_labels, target_names=target_names))
    return acc, test_error


def multi_acc(pred, label, num_classes):
    accs_per_label_pct = []
    tags = torch.argmax(pred, dim=1)
    for c in range(num_classes):  # the three classes
        of_c = label == c
        num_total_per_label = of_c.sum()
        of_c &= tags == label
        num_corrects_per_label = of_c.sum()
        accs_per_label_pct.append(num_corrects_per_label / num_total_per_label * 100)
        return accs_per_label_pct


def test(model, args, epoch, test_loader, test_dataset, optimizer, writer, checkpoint_variables_dict, num_classes):
    """Function to test one epoch with histopathology data and update tensorboard variables

    Args:
        model (torch.nn model) : model for training
        args (argparse) : input arguments containing model settings
        test_dataset (PatchDataset): load patches of test datasets
        test_loader : combines a dataset and a sampler, and provides an iterable over the test dataset.
        epoch (int): epoch number in training
        optimizer (torch.optim.optimizer): optimizer that we are going to use for training
        writer (torch.utils.tensorboard.Writer) : TensorBoard writer which will output to ./runs/ directory by default
        checkpoint_variables_dict (dict) : dict containing all checkpoint variables, values are updated in testing

    Returns :
        checkpoint_variables_dict (dict) : dict containing all checkpoint variables, values are updated in testing

    """
#   best_auc = checkpoint_variables_dict['best_auc']
    best_acc = checkpoint_variables_dict['best_acc']
#   best_confusion = checkpoint_variables_dict['best_confusion']
#   best_prediction_probs = checkpoint_variables_dict['best_prediction_probs']
#   best_auc_slide = checkpoint_variables_dict['best_auc_slide']
#   best_auc_pat = checkpoint_variables_dict['best_auc_pat']
#   best_acc_slide = checkpoint_variables_dict['best_acc_slide']
#   best_acc_pat = checkpoint_variables_dict['best_acc_pat']

    model.eval()
    model_name = 'last_model.pth'
    save_path = os.path.join(args.modeldir, model_name)
    data_to_save = {
        'epoch': epoch+1,
        'state_dict': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        }
    if not args.perform_inference:
        torch.save(data_to_save, save_path)
        print('model saved as %s' % model_name)
    sys.stdout.flush()

    test_loss, test_error = 0., 0.
    true_labels, predicted_labels, prediction_probs, prediction_probs_mat = [], [], [], []
    thenames = []

    # Multi-gpu determines where objective function is stored.
    if args.multigpu:
        calculate_objective = model.module.calculate_objective
    else:
        calculate_objective = model.calculate_objective

    for j in range(args.num_test_repeats):
        mean_loss = 0.0
        prob_vector = []
        attention_map_dict = {}
        for batch_idx, (data, label, names) in enumerate(test_loader):
            bag_label = label[0]
            if j == 0:
                true_labels.append(bag_label.cpu().data.numpy()[0])
                thenames.append(names)

            data, bag_label = data.cuda(), bag_label.cuda()
            data, bag_label = Variable(data), Variable(bag_label)
            #loss, error, predicted_label, prediction_prob, _ = calculate_objective(data, bag_label)
            loss, error, predicted_label, prediction_prob, A = calculate_objective(data, bag_label)

            for i in range(len(names)):
                attention_map_dict[names[i][0]] = float(A[0][i])

            test_loss += loss.data
            test_error += error.data
            prob_vector.append(prediction_prob.cpu().data.numpy()[0])

            del data, label, prediction_prob, loss
            torch.cuda.empty_cache()
            sys.stdout.write("Bags Processed = : %d %f  \r" % (j, batch_idx/len(test_loader)))
            sys.stdout.flush()
        prediction_probs_mat.append(prob_vector)
    predicted_probs = np.mean(prediction_probs_mat, 0)
    predicted_labels = np.argmax(predicted_probs, axis=1)
#   IPython.embed()
#   prediction_probs = np.concatenate(np.mean(prediction_probs_mat, 0),0)
#   predicted_labels = np.zeros(len(prediction_probs))
#   super_threshold_indices = prediction_probs > 0.5
#   predicted_labels[super_threshold_indices] = 1
#   uniqueimgs = [thenames[i][0][0].split('_')[0] for i in range(len(thenames))]
#   consolidate_probs = {k: [] for k in uniqueimgs}
#   consolidate_pats = {}
#   for i in range(len(uniqueimgs)):
#       consolidate_probs[uniqueimgs[i]].append(prediction_probs[i])
#       if uniqueimgs[i][:12] not in consolidate_pats:
#           consolidate_pats[uniqueimgs[i][:12]] = [prediction_probs[i]]
#       else:
#           consolidate_pats[uniqueimgs[i][:12]] = np.append(consolidate_pats[uniqueimgs[i][:12]], prediction_probs[i])
#   true_labels_per_slide, true_labels_per_pat, maxprobs, avgprobs, maxprobspat, avgprobspat = [], [], [], [], [], []
#   seen_img, seen_pat = set(), set()
#   img_list = []
#   for i in range(len(uniqueimgs)):
#       if uniqueimgs[i] not in seen_img:
#           true_labels_per_slide.append(true_labels[i])
#           maxprobs.append(np.max(consolidate_probs[uniqueimgs[i]]))
#           avgprobs.append(np.mean(consolidate_probs[uniqueimgs[i]]))
#           seen_img.add(uniqueimgs[i])
#           img_list.append(uniqueimgs[i])
#       if uniqueimgs[i][:12] not in seen_pat:
#           true_labels_per_pat.append(true_labels[i])
#           maxprobspat.append(np.max(consolidate_pats[uniqueimgs[i][:12]]))
#           avgprobspat.append(np.mean(consolidate_pats[uniqueimgs[i][:12]]))
#           seen_pat.add(uniqueimgs[i][:12])
#   maxprobs = np.array(maxprobs)
#   avgprobs = np.array(avgprobs)
#   maxprobspat = np.array(maxprobspat)
#   avgprobspat = np.array(avgprobspat)
#   true_labels_per_slide = np.array(true_labels_per_slide)
#   true_labels_per_pat = np.array(true_labels_per_pat)

    #np.save('attention_map_dict.npy', attention_map_dict)
    np.save(os.path.join(args.modeldir, 'attention_map_dict.npy'), attention_map_dict)

    '''
    if args.perform_inference:
        with open(os.path.join(args.modeldir,'predicted_files.txt'), 'w') as f:
            for item in img_list:
                f.write("%s\n" % item)
#       np.save(os.path.join(args.modeldir,'predictions.npy'), maxprobs)
        np.save(os.path.join(args.modeldir,'predictions.npy'), predicted_probs)
#       np.save(os.path.join(args.modeldir,'labels.npy'), true_labels_per_slide)
        np.save(os.path.join(args.modeldir,'labels.npy'), np.array(true_labels))
    '''
    test_loss /= (len(test_dataset)*args.num_test_repeats)
    test_error /= (len(test_dataset)*args.num_test_repeats)

    acc, test_error = metrics(predicted_probs, true_labels)

    # accs_per_label_pct = multi_acc(prediction_probs, true_labels, num_classes)
    # print(accs_per_label_pct)

#   acc, test_error = metrics(prediction_probs, true_labels)
#   acc, test_error, auc_score, confusion = metrics(prediction_probs, true_labels)
#   acc_slide_avg, test_error_slide_avg, auc_score_slide_avg, confusion_slide_avg = metrics(avgprobs, true_labels_per_slide)
#   acc_slide_max, test_error_slide_max, auc_score_slide_max, confusion_slide_max = metrics(maxprobs, true_labels_per_slide)
#   acc_pat_avg, test_error_pat_avg, auc_score_pat_avg, confusion_pat_avg = metrics(avgprobspat, true_labels_per_pat)
#   acc_pat_max, test_error_pat_max, auc_score_pat_max, confusion_pat_max = metrics(maxprobspat, true_labels_per_pat)


#   if auc_score_slide_max > best_auc_slide and not args.perform_inference:
#       best_prediction_probs = prediction_probs
#       best_confusion = confusion
#       model_name = 'best_auc_model.pth'

#       save_path = os.path.join(args.modeldir, model_name)
#       data_to_save = {
#           'epoch': epoch+1,
#           'state_dict': model.state_dict(),
#           'optimizer_state': optimizer.state_dict(),
#           'test_error': test_error,
#           'auc': auc_score,
#           'best_auc_slide': auc_score_slide_max,
#           'auc_pat': auc_score_pat_max,
#           'confusion_matrix': confusion,
#           'confusion_matrix_slide': confusion_slide_max,
#           'confusion_matrix_pat': confusion_pat_max,
#           'acc': acc,
#           'prediction_probs': prediction_probs,
#           'true_labels': true_labels,
#           'thenames': thenames
#           }
#       torch.save(data_to_save, save_path)
#       print('best auc model saved as %s' % model_name)
#       sys.stdout.flush()
#   best_auc = max(best_auc, auc_score)
#   best_auc_slide = max(best_auc_slide, auc_score_slide_max)
#   best_auc_pat = max(best_auc_pat, auc_score_pat_max)
    best_acc = max(best_acc, acc)
#   best_acc_slide = max(best_acc_slide, acc_slide_max)
#   best_acc_pat = max(best_acc_pat, acc_pat_max)

    writer.add_scalar('loss_test', test_loss, epoch)
    writer.add_scalar('error_test', test_error, epoch)
    # writer.add_scalar('balanced_accuracy_test', bacc, epoch)
#   writer.add_scalar('auc_test', auc_score, epoch)
#   writer.add_scalar('mean_prediction_prob_test', np.mean(prediction_probs), epoch)
#   writer.add_scalar('mean_prediction_prob_test', np.mean(predicted_probs), epoch)

#   writer.add_scalar('acc_slide_avg', acc_slide_avg, epoch)
#   writer.add_scalar('auc_score_slide_avg', auc_score_slide_avg, epoch)
#   writer.add_scalar('acc_slide_max', acc_slide_max, epoch)
#   writer.add_scalar('auc_score_slide_max', auc_score_slide_max, epoch)
#   writer.add_scalar('acc_pat_avg', acc_pat_avg, epoch)
#   writer.add_scalar('auc_score_pat_avg', auc_score_pat_avg, epoch)
#   writer.add_scalar('acc_pat_max', acc_pat_max,epoch)
#   writer.add_scalar('auc_score_pat_max', auc_score_pat_max, epoch)

#   print('\nTest Set, Test AUC Complete: {:.4f}, Loss: {:.4f}, Test error: {:.4f}, Test AUC: {:.4f}'
#         .format(auc_score_slide_max, test_loss.cpu().numpy()[0], test_error, auc_score))
#   print('\nConfusion matrix :', confusion)
    print('\nTest Set Complete: Loss: {:.4f}, Test error: {:.4f}'
          .format(test_loss.cpu().numpy(), test_error))

#   checkpoint_variables_dict['best_auc'] = best_auc
    checkpoint_variables_dict['best_acc'] = best_acc
#   checkpoint_variables_dict['best_confusion'] = best_confusion
#   checkpoint_variables_dict['best_prediction_probs'] = best_prediction_probs
#   checkpoint_variables_dict['best_auc_slide'] = best_auc_slide
#   checkpoint_variables_dict['best_auc_pat'] = best_auc_pat
#   checkpoint_variables_dict['best_acc_slide'] = best_acc_slide
#   checkpoint_variables_dict['best_acc_pat'] = best_acc_pat

    return checkpoint_variables_dict
