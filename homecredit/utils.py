"""Utility functions."""

import numpy as np
import pandas as pd

def weighted_rank_average(df: pd.DataFrame, weights: np.array) -> np.array:
    """
    Compute the weighted rank average of predictions.

    Arguments:
        df: pandas dataframe with predictions
        weights: numpy array with weights for each column in the dataframe

    Returns:
        numpy array with the weighted rank average.
    """
    
    # Check if weights sum to 1, if not normalize them
    weights = weights / weights.sum()
    
    # Convert probabilities to ranks
    ranks = df.rank(method='average')
    
    # Apply weights to ranks
    weighted_ranks = ranks.multiply(weights).values
    
    # Compute the sum of the weighted ranks along the rows
    weighted_average_ranks = weighted_ranks.sum(axis=1)
    
    # Normalize the weighted average of ranks to be between 0 and 1
    min_rank = weighted_average_ranks.min()
    max_rank = weighted_average_ranks.max()
    normalized_probabilities = (weighted_average_ranks - min_rank) / (max_rank - min_rank)
    
    return normalized_probabilities