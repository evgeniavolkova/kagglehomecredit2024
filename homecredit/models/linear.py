"""Linear models."""

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

class LogReg(BaseEstimator, ClassifierMixin):
    """Logistic Regression model."""

    def __init__(self, params: dict) -> None:
        """
        Initialize the Logistic Regression model.

        Arguments:
            params: Parameters for the model.
        """
        self.params = params
        self.model = LogisticRegression(**params)
        self.features = None
        self.classes_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: bool = False) -> None:
        """
        Fit the model.

        Arguments:
            X: Features to fit on.
            y: Target variable.
            verbose: If True, will print out information about the fitting process.
        """
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        self.features = X.columns
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.array:
        """
        Predict probabilities.

        Arguments:
            X: Features to predict on.
        Returns:
            np.array: Predicted probabilities.
        """
        return self.model.predict_proba(X)

    def set_seed(self, seed: int) -> None:
        """
        Set the seed for the model.

        Arguments:
            seed: Seed to set.
        """
        self.params["random_state"] = seed

    def get_params(self, deep: bool = True) -> dict:
        """
        Get parameters for this estimator.

        Arguments:
            deep: If True, will return the parameters for this
                  estimator and contained subobjects that are estimators.
        """
        return {
            "params": self.params,
        }

    def set_params(self, **parameters):
        """
        Set the parameters of this estimator.

        Arguments:
            parameters: A dictionary of parameters to set, mapping parameter
                        names to their new values.
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
