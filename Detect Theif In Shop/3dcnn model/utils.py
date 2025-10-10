# utils.py
import os
import random
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def compute_class_weights(dataset):
    # dataset: instance of VideoDataset or list of (path,label)
    counts = {}
    for _, label in getattr(dataset, "items", []):
        counts[label] = counts.get(label, 0) + 1
    # produce weight per class for CrossEntropyLoss (inverse freq)
    labels = sorted(counts.keys())
    freqs = [counts[l] for l in labels]
    total = sum(freqs)
    weights = [total / f for f in freqs]
    # map to tensor where index corresponds to class label
    # assume labels are 0..K-1 contiguous
    max_label = max(labels) if labels else 0
    weight_tensor = torch.ones(max_label + 1)
    for i, l in enumerate(labels):
        weight_tensor[l] = weights[i]
    return weight_tensor

def metrics_from_preds(labels_true, labels_pred):
    # returns dict with accuracy, precision, recall, f1
    acc = accuracy_score(labels_true, labels_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(labels_true, labels_pred, average="binary", zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
