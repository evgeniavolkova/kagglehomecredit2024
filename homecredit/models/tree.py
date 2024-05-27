"""
Tree-based models.
Enhanced versions of LightGBM, CatBoost, and XGBoost models with additional functionality.
"""

import gc

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split

import lightgbm as lgb
import xgboost as xgb
import catboost as cb

from ..config import RANDOM_SEED


class LGBM(BaseEstimator, ClassifierMixin):
    """LightGBM model."""

    def __init__(
            self, 
            params: dict, 
            early_stopping: bool = True, 
            early_stopping_rounds: int = 50, 
            test_size: float = 0.05, 
            shuffle: bool = False, 
            smote: bool = False, 
            neighbors: int = 0, 
            weight_min: float = 0.0
        ) -> None:
        """
        Initialize the LGBM model.

        Arguments:
            params: Dictionary with parameters for the model.
            early_stopping: Whether to use early stopping.
            early_stopping_rounds: Number of rounds for early stopping.
            test_size: Size of the test set.
            shuffle: Whether to shuffle the data.
            smote: Whether to use SMOTE. Not implemented, kept for compatibility.
            neighbors: Number of neighbors for KNN. Not implemented, kept for compatibility.
            weight_min: Minimum weight. Not implemented, kept for compatibility.
        """
        self.params = params
        self.early_stopping = early_stopping
        self.early_stopping_rounds = early_stopping_rounds
        self.test_size = test_size
        self.shuffle = shuffle
        self.smote = smote
        self.neighbors = neighbors
        self.weight_min = weight_min

        self.model = lgb.LGBMClassifier(**params)

        self.features = None
        self.classes_ = None
        self.nn_k = None
        self.train_targets = None

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: bool = False) -> None:
        """
        Fit the model.

        Arguments:
            X: DataFrame with features.
            y: Series with target.
            verbose: Whether to print the results.
        """
        if not self.early_stopping:
            X_train, y_train = X, y

            eval_set = [(X_train, y_train)]
            callbacks = [
                lgb.callback.log_evaluation(50 if verbose else 0),
            ]
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.test_size, 
                shuffle=self.shuffle, 
                random_state=RANDOM_SEED if self.shuffle else None
            )

            eval_set = [(X_train, y_train), (X_val, y_val)]
            callbacks = [
                lgb.callback.log_evaluation(50 if verbose else 0),
                lgb.callback.early_stopping(
                    stopping_rounds=self.early_stopping_rounds,
                    verbose=verbose
                )
            ]

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            callbacks=callbacks
        )
        self.classes_ = self.model.classes_
        self.features = self.model.feature_name_
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.array:
        """
        Predict probabilities.

        Arguments:
            X: DataFrame with features.
        Returns:
            np.array: Predicted probabilities.
        """
        return self.model.predict_proba(X)

    def get_feature_importances(self) -> None:
        """
        Get feature importances.

        Returns:
            DataFrame with feature importances.
        """
        imp_df = pd.DataFrame({
            "feature": self.features,
            "imp": self.model.feature_importances_
        })
        imp_df.sort_values("imp", ascending=False, inplace=True)
        imp_df["imp_acc"] = imp_df["imp"].cumsum() / imp_df["imp"].sum()
        imp_df.set_index("feature", inplace=True)
        return imp_df

    def set_seed(self, seed: int) -> None:
        """
        Set the seed for the model.

        Arguments:
            seed: Seed to set.
        """
        self.params["random_state"] = seed

    def get_params(self, deep: bool = True):
        """
        Get parameters for this estimator.

        Arguments:
            deep: If True, will return the parameters for this
                  estimator and contained subobjects that are estimators.
        """
        return {
            "params": self.params,
            "early_stopping": self.early_stopping,
            "early_stopping_rounds": self.early_stopping_rounds,
            "test_size": self.test_size,
            "shuffle": self.shuffle,
            "smote": self.smote,
            "neighbors": self.neighbors,
            "weight_min": self.weight_min,
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
    


class CBM(BaseEstimator, ClassifierMixin):
    """CatBoost model."""

    def __init__(
            self, 
            params: dict, 
            early_stopping: bool = True, 
            early_stopping_rounds: int = 50, 
            test_size: float = 0.05, 
            shuffle: bool = False,
            smote: bool = False, 
            neighbors: int = 0,
            sample_weights: bool = False,
        ) -> None:
        """
        Initialize the CatBoost model.

        Arguments:
            params: Dictionary with parameters for the model.
            early_stopping: Whether to use early stopping.
            early_stopping_rounds: Number of rounds for early stopping.
            test_size: Size of the test set.
            shuffle: Whether to shuffle the data.
            smote: Whether to use SMOTE. Not implemented, kept for compatibility.
            neighbors: Number of neighbors for KNN. Not implemented, kept for compatibility.
            sample_weights: sample weights. Not implemented, kept for compatibility.
        """
        self.params = params
        self.early_stopping = early_stopping
        self.early_stopping_rounds = early_stopping_rounds
        self.test_size = test_size
        self.shuffle = shuffle
        self.sample_weights = sample_weights

        self.model = cb.CatBoostClassifier(**params, early_stopping_rounds=early_stopping_rounds)

        self.cat_cols = None
        self.features = None
        self.classes_ = None
        self.nn_k = None
        self.train_targets = None

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: bool = False) -> None:
        """
        Fit the model.

        Arguments:
            X: DataFrame with features.
            y: Series with target.
            verbose: Whether to print the results.
        """
        cat_cols = X.select_dtypes(include=["category"]).columns.tolist()
        self.cat_cols = cat_cols
        if self.early_stopping:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.test_size,
                shuffle=self.shuffle,
                random_state=RANDOM_SEED if self.shuffle else None
            )
            X_train[cat_cols] = X_train[cat_cols].astype(str)
            X_val[cat_cols] = X_val[cat_cols].astype(str)

            train_pool = cb.Pool(X_train, y_train,cat_features=cat_cols)
            val_pool = cb.Pool(X_val, y_val, cat_features=cat_cols)

            eval_set = val_pool
        else:
            X_train, y_train = X, y
            X_train[cat_cols] = X_train[cat_cols].astype(str)
            train_pool = cb.Pool(X_train, y_train,cat_features=cat_cols)
            eval_set = None

        self.model.fit(train_pool, eval_set=eval_set, verbose=50 if verbose else 0)
        self.classes_ = self.model.classes_
        self.features = self.model.feature_names_
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.array:
        """
        Predict probabilities.

        Arguments:
            X: DataFrame with features.
        Returns:
            np.array: Predicted probabilities.
        """
        X[self.cat_cols] = X[self.cat_cols].astype(str)
        return self.model.predict_proba(X)

    def set_seed(self, seed: int) -> None:
        """
        Set the seed for the model.

        Arguments:
            seed: Seed to set.
        """
        self.params["random_seed"] = seed

    def get_params(self, deep: bool = True):
        """
        Get parameters for this estimator.

        Arguments:
            deep: If True, will return the parameters for this
                  estimator and contained subobjects that are estimators.
        """
        return {
            "params": self.params,
            "early_stopping": self.early_stopping,
            "early_stopping_rounds": self.early_stopping_rounds,
            "test_size": self.test_size,
            "shuffle": self.shuffle,
            "sample_weights": self.sample_weights,
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
    
class XGBM(BaseEstimator, ClassifierMixin):
    """XGBoost model."""

    def __init__(
            self, 
            params: dict, 
            early_stopping: bool = True, 
            early_stopping_rounds=50, 
            test_size: float = 0.05,
            shuffle: bool = False
        ):
        """
        Initialize the XGBM model.

        Arguments:
            params: Dictionary with parameters for the model.
            early_stopping: Whether to use early stopping.
            early_stopping_rounds: Number of rounds for early stopping.
            test_size: Size of the test set.
            shuffle: Whether to shuffle the data.
        """    
        self.params = params
        self.early_stopping = early_stopping
        self.early_stopping_rounds = early_stopping_rounds
        self.model = xgb.XGBClassifier(**params, early_stopping_rounds=early_stopping_rounds)
        self.test_size = test_size
        self.shuffle = shuffle
        self.features = None
        self.classes_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: bool = False) -> None:
        """
        Fit the model.

        Arguments:
            X: DataFrame with features.
            y: Series with target.
            verbose: Whether to print the results.
        """
        if not self.early_stopping:
            X_train, y_train = X, y

            eval_set = [(X_train, y_train)]
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.test_size, 
                shuffle=self.shuffle, 
                random_state=RANDOM_SEED if self.shuffle else None
            )

            eval_set = [(X_train, y_train), (X_val, y_val)]

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose= 50 if verbose else 0,
        )
        self.classes_ = self.model.classes_
        self.features = X.columns
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.array:
        """
        Predict probabilities.

        Arguments:
            X: DataFrame with features.
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

    def get_params(self, deep: bool = True):
        """
        Get parameters for this estimator.

        Arguments:
            deep: If True, will return the parameters for this
                  estimator and contained subobjects that are estimators.
        """
        return {
            "params": self.params,
            "early_stopping": self.early_stopping,
            "early_stopping_rounds": self.early_stopping_rounds,
            "test_size": self.test_size,
            "shuffle": self.shuffle,
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
    