import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score
from scipy.ndimage import gaussian_filter


def gaussian_smooth(x, sigma=4):
    bs = x.shape[0]
    for i in range(0, bs):
        x[i] = gaussian_filter(x[i], sigma=sigma)
    return x


def rescale(x):
    return (x - x.min()) / (x.max() - x.min())


def get_threshold(gt, score):
    gt_mask = np.asarray(gt)
    precision, recall, thresholds = precision_recall_curve(gt_mask.flatten(), score.flatten())
    a = 2 * precision * recall
    b = precision + recall
    f1 = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    threshold = thresholds[np.argmax(f1)]
    return threshold


def roc_auc_img(gt, score):
    img_roc_auc = roc_auc_score(gt, score)

    return img_roc_auc


def roc_auc_pxl(gt, score):
    per_pxl_roc_auc = roc_auc_score(gt.flatten(), score.flatten())

    return per_pxl_roc_auc


def cal_img_roc(scores, gt_list):
    img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
    gt_list = np.asarray(gt_list)
    fpr, tpr, _ = roc_curve(gt_list, img_scores)
    img_roc_auc = roc_auc_img(gt_list, img_scores)

    return fpr, tpr, img_roc_auc


def cal_pxl_roc(gt_mask, scores):
    fpr, tpr, _ = roc_curve(gt_mask.flatten(), scores.flatten())
    per_pxl_rocauc = roc_auc_pxl(gt_mask.flatten(), scores.flatten())

    return fpr, tpr, per_pxl_rocauc
