"""Wrapper for scikit-learn's Pipeline with additional functionalities."""

import gc
import joblib
import os

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sklearn.base import clone
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from .config import PATH_MODELS, COL_TARGET, COL_ID, COL_DATE, COL_WEEK, RANDOM_SEED
from .metrics import gs_metric
from .data import utils

class FullPipeline:
    """
    Enhances scikit-learn's Pipeline by adding functionalities to manage models.
    """
    def __init__(
            self,
            model: Pipeline,
            run_name: str = "",
            name: str = "",
            load_model: bool = False,
            features: list | None = None,
            save_to_disc: bool = True,
            target_col: str = COL_TARGET
        ) -> None:
        """
        Initialize.

        Arguments:
            model: The scikit-learn pipeline to be used.
            name: Name of the model for saving and loading purposes.
            load_model: Whether to load the model from disk.
            features: List of feature names to be used. If None, all features will be considered.
            save_to_disc: Whether to save the model to disk.
            target_col: Name of the target column.
        """
        self.model = model
        self.name = name
        self.load_model = load_model
        self.features = features
        self.save_to_disc = save_to_disc
        self.target_col = target_col

        self.set_run_name(run_name)
        self.path = os.path.join(PATH_MODELS, f"{self.run_name}")

    def set_run_name(self, run_name: str) -> None:
        """
        Sets the run name for the model.

        Arguments:
            run_name: The name of the run.
        """
        self.run_name = run_name
        self.path = os.path.join(PATH_MODELS, f"{self.run_name}")
        utils.create_folder(self.path)

    def fit(self, df: pd.DataFrame | None = None, verbose: bool = False) -> None:
        """
        Fit the model pipeline.

        Arguments:
            df: The DataFrame containing the data to be used for model fitting.
            verbose: Enables verbose output during model fitting.
        """
        if not self.load_model:
            if self.features is None:
                self.features = [i for i in df.columns if i not in [self.target_col, COL_TARGET, COL_ID, COL_DATE, COL_WEEK]]

            X_train = df[self.features].copy()
            y_train = df[self.target_col]

            self.model.fit(X_train, y_train, classifier__verbose=verbose)
            if self.save_to_disc:
                self.save()
        else:
            self.load()
            if self.features is None:
                self.features = self.model.named_steps["classifier"].features
        if verbose:
            print(f"Number of features: {len(self.features)}")

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities.

        Arguments:
            df: The DataFrame containing the data to be used for prediction.
        Returns:
            preds: array with predicted probabilities
        """
        X_valid = df[self.features].copy()
        preds = self.model.predict_proba(X_valid)[:,1]

        return preds

    def predict_proba_in_batches(self, df: pd.DataFrame, batch_size: int = 2**15) -> np.ndarray:
        """
        Predict probabilities in batches.

        Arguments:
            df: The DataFrame containing the data to be used for prediction.
            batch_size: The batch size for prediction.
        Returns:
            preds: array with predicted probabilities
        """
        X_valid = df[self.features]

        n = len(X_valid)
        n_batches = int(np.ceil(n / batch_size))
        preds = np.zeros((n,))

        for batch_idx in tqdm(range(n_batches)):
            idx1 = batch_idx * batch_size
            idx2 = min((batch_idx + 1) * batch_size, n)
            batch_preds = self.model.predict_proba(X_valid.iloc[idx1:idx2])[:,1]
            preds[idx1:idx2] = batch_preds
            gc.collect()

        return preds

    def load(self) -> None:
        """
        Load the model.
        """
        self.model = joblib.load(f"{self.path}/model_{self.name}.joblib")

    def save(self) -> None:
        """
        Save the model.
        """
        joblib.dump(self.model, f"{self.path}/model_{self.name}.joblib")

    def get_params(self, deep: bool = True) -> dict:
        """
        Get parameters for this estimator.

        Arguments:
            deep: If True, will return the parameters for this
                  estimator and contained subobjects that are estimators.
        """
        return {
            "model": self.model,
            "name": self.name,
            "load_model": self.load_model,
            "features": self.features,
            "save_to_disc": self.save_to_disc,
            "target_col": self.target_col
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

    def get_feature_importances(self) -> pd.DataFrame:
        """
        Get feature importances from the model.

        Returns:
            DataFrame with feature importances.
        """
        return self.model.named_steps["classifier"].get_feature_importances()
    
class PipelineCV:
    """
    Cross-validation pipeline.
    """
    def __init__(
            self, 
            model: FullPipeline, 
            n_splits: int, 
            reverse: bool = False, 
            weights: bool = False, 
            shuffle: bool = False, 
            ts: bool = False
        ) -> None:
        """
        Initialize.

        Arguments:
            model: FullPipeline model.
            n_splits: Number of splits.
            reverse: Whether to reverse the order of the splits.
            weights: Whether to use weights.
            shuffle: Whether to shuffle the data.
            ts: Whether to use time series split.
        """
        self.model = model
        self.n_splits = n_splits
        self.reverse = reverse
        self.weights = weights
        self.shuffle = shuffle
        self.ts = ts
        self.models = []

    def fit(self, df: pd.DataFrame, verbose: bool = False, predict: bool = True) -> None:
        """
        Fit models.

        Arguments:
            df: DataFrame with data.
            verbose: Whether to print verbose output.
            predict: Whether to return OOF predictions.
        """
        preds = np.zeros(len(df)) + np.nan
        preds_df = df[[COL_ID, COL_WEEK, COL_TARGET]].copy()
        preds_df["preds"] = np.nan

        weeks_unique = df["WEEK_NUM"].sort_values().unique()
        if self.ts:
            cv = TimeSeriesSplit(n_splits=self.n_splits, test_size=15)
        else:
            cv = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=RANDOM_SEED if self.shuffle else None)
        cv_split = cv.split(weeks_unique)
        for fold, (train_idx, valid_idx) in enumerate(cv_split):
            if verbose:
                print("-"*20 + f"Fold {fold}" + "-"*20)

            weeks_train = weeks_unique[train_idx]
            weeks_valid = weeks_unique[valid_idx]

            train_idx = df["WEEK_NUM"].isin(weeks_train)
            valid_idx = df["WEEK_NUM"].isin(weeks_valid)

            df_train = df.loc[train_idx]
            df_valid = df.loc[valid_idx]

            # Fit model
            model_fold = clone(self.model)
            model_fold.set_run_name(f"fold{fold}")
            model_fold.fit(df_train, verbose=verbose)
            self.models.append(model_fold)

            # Preds
            if predict:
                preds = model_fold.predict_proba(df_valid)
                preds_df.loc[valid_idx, "preds"] = preds
                score, ginis_dict = gs_metric(df_valid[COL_TARGET].values, preds, df_valid["WEEK_NUM"].values, penalty=False, verbose=False)
                print(
                    f"Stability gini: {score:.3f}"
                    f", gini: {np.mean(ginis_dict['ginis']):.3f}"
                    f", slope: {ginis_dict['slope']:.3f}"
                    f", std: {ginis_dict['std']:.3f}"
                )
        preds_df.dropna(inplace=True)
        return preds_df

    def load(self) -> None:
        """
        Load models.
        """
        self.models = []
        for i in range(self.n_splits):
            model = clone(self.model)
            model.set_run_name(f"fold{i}")
            model.fit()
            self.models.append(model)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities.

        Arguments:
            df: DataFrame with data.
        Returns:
            preds: array with predicted probabilities.
        """
        preds = []
        for i in range(self.n_splits):
            preds.append(self.models[i].predict_proba(df))

        preds = np.stack(preds, axis=1)
        preds = np.mean(preds, axis=1)
        return preds

    def predict_proba_in_batches(self, df: pd.DataFrame, batch_size: int = 2**15) -> np.ndarray:
        """
        Predict probabilities in batches.

        Arguments:
            df: DataFrame with data.
            batch_size: Batch size for prediction.
        Returns:
            preds: array with predicted probabilities.
        """
        preds = []
        for i in range(self.n_splits):
            preds.append(self.models[i].predict_proba_in_batches(df, batch_size))
        preds = np.stack(preds, axis=1)
        preds = np.mean(preds, axis=1)
        return preds

    def get_feature_importances(self) -> pd.DataFrame:
        """
        Get feature importances.

        Returns:
            DataFrame with feature importances.
        """
        imp_ls = []
        for i in range(self.n_splits):
            imp_df = self.models[i].get_feature_importances()
            imp_ls.append(imp_df["imp"])

        imp_df = pd.concat(imp_ls, axis=1)
        folds = [i for i in range(self.n_splits)]
        imp_df.columns = folds
        imp_df["imp"] = imp_df[folds].mean(axis=1)
        imp_df["spread"] = imp_df[folds].max(axis=1)-imp_df[folds].min(axis=1)
        imp_df.sort_values("imp", ascending=False, inplace=True)
        imp_df["imp_acc"] = imp_df["imp"].cumsum() / imp_df["imp"].sum()
        return imp_df