from typing import List, Optional, Callable, Union, Any, Tuple

import torch
import numpy as np
from torch import Tensor
from numpy import ndarray
from scipy.stats import pearsonr, spearmanr

from .metrics import concordance_index_censored as ci

from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)
from scipy.special import softmax, expit 

def concordance_index(
    y_true: Union[Tensor, ndarray], 
    y_pred: Union[Tensor, ndarray],
) -> float:
    """Compute the concordance-index value.

    For coxph model
    Args:
        y_true (Union[Tensor, ndarray]): Observed time. Negative values are considered right censored.
        y_pred (Union[Tensor, ndarray]): Predicted value (proportional hazard).

    For discrete model
    Args:
        y_true (Union[Tensor, ndarray]): Observed time (at the first column) and censorship (at the second column). 
        y_pred (Union[Tensor, ndarray]): Predicted value (time-dependent hazard function).
    """
    if isinstance(y_pred, Tensor):
        y_pred = y_pred.numpy()
    if isinstance(y_true, Tensor):
        y_true = y_true.numpy()

    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    t = np.abs(y_true)
    e = (y_true > 0).astype(bool)
    return ci(e, t, -y_pred, tied_tol=1e-08)[0]


def evaluator(y, y_hat, metrics='cindex', **kws):
    """
    If it is a discrete model:
        y: [B, 2] (col1: y_t, col2: y_c)
        y_hat: [B, BINS]
    else:
        y: [B, 1]
        y_hat: [B, 1]
    """

    if metrics == 'cindex':
        return {'ci': concordance_index(y, y_hat)}
    elif metrics == 'classification':
        y_hat = np.squeeze(y_hat)
        y_prob = expit(y_hat)  # sigmoid
        y_pred = (y_prob >= 0.5).astype(int)
        y_true = np.array(y).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # recall
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'sensitivity': sensitivity,
            'specificity': specificity,
            'auc': roc_auc_score(y_true, y_prob),
        }
        return metrics
    elif metrics == 'regression':
        y_squeezed = y.squeeze()
        y_hat_squeezed = y_hat.squeeze()
        pearson_corr, _ = pearsonr(y_squeezed, y_hat_squeezed)
        spearman_corr, _ = spearmanr(y_squeezed, y_hat_squeezed)
        
        return {
            'mse': mean_squared_error(y, y_hat),
            'mae': mean_absolute_error(y, y_hat),
            'r2': r2_score(y, y_hat),
            'pearson': pearson_corr,
            'spearman': spearman_corr
        }
    else:
        raise NotImplementedError(f"Metrics {metrics} has not implemented.")
