"""General Helper functions.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

def local_excel_files(root_dir):
    """Returns filenames to excel files contained in the directory tree of root_dir
    """
    excel_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if ".xlsx" in file:
                excel_files.append(os.path.join(root, file))
    return excel_files


def sens_spec(gt, preds):
    """Assume value of 1 for sens, value of 0 for spec
    """
    classes = np.unique(gt)
    assert len(classes) == 2
    sens = np.sum(preds[gt == 1] == gt[gt == 1]) / len(gt[gt ==1])
    spec = np.sum(preds[gt == 0] == gt[gt == 0]) / len(gt[gt ==0])
    return sens, spec


def plot_ss_curve(gt, probs, title=""):
    fpr, tpr, thresholds = metrics.roc_curve(gt, probs, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    plt.plot(1-fpr, tpr)
    plt.title(title)
    plt.xlabel("Specificity")
    plt.ylabel("Sensitivity")
    plt.show()
    print("AUC={:0.3f}".format(auc))
    acc_bal = max([np.mean((ss, sp)) for ss, sp in zip(tpr, 1-fpr)])
    print("Best balanced accuracy: {:0.3f}".format(acc_bal))


def accuracy_score_balanced(ground_truth, predictions):
    """Returns the individual class sensitivites, as well as their mean

    E.g.
    gt = [0] * 90 + [1] * 10
    preds = [0] * 100

    # Prints 0.5 accuracy - chance.
    ab, pca = accuracy_score_balanced(gt, preds)
    print("Balanced Accuracy: {}".format(ab))
    print("Per_class_accuracy: {}".format(pca))

    # Prints 0.9 accuracy - misleading.
    print("Unbalanced accuracy: {}".format(accuracy_score(gt, preds)))

	Args:
		ground_truth(array-like): y
		predictions(array-like): yhat

	Returns:
        acc_balanced (float): The mean of the individual class sensitivies.
        per_class_accuracies(dict): Format of {class_index : class_sensitivity}
    """
    assert len(ground_truth) == len(predictions)
    gt = np.array(ground_truth)
    preds = np.array(predictions)
    per_class_accuracies = {}
    for class_ in np.unique(gt):
        sens = np.sum(preds[gt == class_] == gt[gt == class_]) / np.sum(gt == class_)
        per_class_accuracies[class_] = sens
    acc_balanced = np.mean(list(per_class_accuracies.values()))
    return acc_balanced, per_class_accuracies


