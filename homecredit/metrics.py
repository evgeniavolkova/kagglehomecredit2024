"""Gini stability metric."""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


def gs_metric(
        true: np.ndarray,
        pred: np.ndarray,
        week: np.ndarray,
        penalty: bool = False,
        verbose: bool = False,
) -> float:
    """
    Calculate the Gini Stability metric.

    See details here:
    www.kaggle.com/competitions/home-credit-credit-risk-model-stability/overview/evaluation

    Arguments:
        true: array with labels
        pred: array with predictions
        week: array with week numbers
        penalty: whether to apply a slope penalty
        verbose: whether to print the results
    Returns:
        out: float
    """
    # Sort by week
    sorted_indices = np.argsort(week)
    week_sorted = week[sorted_indices]
    true_sorted = true[sorted_indices]
    pred_sorted = pred[sorted_indices]

    # Group by week
    week_unique, week_index = np.unique(week_sorted, return_index=True)
    grouped_true = np.split(true_sorted, week_index[1:])
    grouped_pred = np.split(pred_sorted, week_index[1:])

    # Calculate Gini for each week
    ginis = np.zeros(len(week_unique))
    for i, (true, pred) in enumerate(zip(grouped_true, grouped_pred)):
        if len(np.unique(true)) == 1:
            gini = 0.0
        else:
            gini = roc_auc_score(true, pred) * 2 - 1
        ginis[i] = gini

    # Calculate Gini Stability
    slope, intercept, _, _, _ = stats.linregress(week_unique, ginis)
    residuals = ginis - (slope*week_unique + intercept)
    out = np.mean(ginis) - 0.5 * np.std(residuals)
    if penalty:
        out += 88.0 * min(0, slope)

    # Print results
    if verbose:
        print(
            f"Stability gini: {out:.3f}"
            f", gini: {np.mean(ginis):.3f}"
            f", slope: {88.0 * slope:.3f}"
            f", std: {0.5 * np.std(residuals):.3f}"
        )
        plt.plot(ginis)

    out_dict = {
        "ginis": ginis,
        "slope": 88.0*slope,
        "std": np.std(residuals)
    }

    return out, out_dict