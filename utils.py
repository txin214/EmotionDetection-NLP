import os
import random
import logging
import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics import hamming_loss

'''
Modified from https://github.com/monologg/GoEmotions-pytorch/blob/master/utils.py
'''

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(labels, preds):
    assert len(preds) == len(labels)
    results = dict()
    # Accuracy gives percentage of exact matches
    results["accuracy"] = accuracy_score(labels, preds)
    results["hamming_loss"] = hamming_loss(labels, preds)
    results["macro_precision"], results["macro_recall"], results[
        "macro_f1"], _ = precision_recall_fscore_support(
        labels, preds, average="macro")
    results["micro_precision"], results["micro_recall"], results[
        "micro_f1"], _ = precision_recall_fscore_support(
        labels, preds, average="micro")
    results["weighted_precision"], results["weighted_recall"], results[
        "weighted_f1"], _ = precision_recall_fscore_support(
        labels, preds, average="weighted")

    return results


#### For calculating CPCC. Reference: https://openreview.net/pdf?id=7J-30ilaUZM

def tree_metric(label1, label2):
    if label1.label == label2.label:
        return 0
    if label1.ekman == label2.ekman:
        return 1
    if label1.sentiment == label2.sentiment:
        return 2
    return 3

