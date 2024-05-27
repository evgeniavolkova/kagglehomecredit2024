"""Ensemble model selection."""

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from homecredit.config import COL_ID, COL_DATE, COL_WEEK, COL_TARGET
from homecredit.metrics import gs_metric

class EnsembleSelector:
    """Select models for the ensemble."""

    def __init__(self, params: dict, method: str = "score") -> None:
        """
        Initialize the EnsembleSelector.

        Arguments:
            params: Dictionary with parameters for the ensemble selection.
            method: Method for the ensemble selection: "score", "corr", "forward".
        """
        self.params = params
        self.method = method
        self.methods_map = {
            "score": self.find_best_ensemble_score,
            "corr": self.find_best_ensemble_corr,
            "forward": self.find_best_ensemble_forward,
        }
        self.find_best_ensemble = self.methods_map[self.method]
        self.selected_models = None

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: bool = True) -> None:
        """
        Fit the ensemble selector.

        Arguments:
            X: Features to fit on.
            y: Target variable.
            verbose: If True, will print out information about the fitting process.
        """
        self.selected_models = self.find_best_ensemble(X, y)

    def predict_proba(self, X: pd.DataFrame) -> np.array:
        """
        Predict probabilities.

        Arguments:
            X: Features to predict on.
        Returns:
            np.array: Predicted probabilities.
        """
        preds = X[self.selected_models].mean(axis=1).values
        preds = preds[:, np.newaxis]
        preds = np.hstack((1-preds, preds))
        return preds

    def find_best_ensemble_score(
            self, 
            X: pd.DataFrame, 
            y: pd.Series, 
            corr_threshold: float = 0.95, 
            max_n: int = 10
        ) -> list:
        """
        Find the best combination of models.
        Iteratively adds models to the ensemble based on the score with a threshold for correlation.

        Arguments:
            X: Features to fit on.
            y: Target variable.
            corr_threshold: Threshold for correlation.
            max_n: Maximum number of models to add.
        Returns:
            list: List of selected models.
        """
        if "corr_threshold" in self.params:
            corr_threshold = self.params["corr_threshold"]
        if "max_n" in self.params:
            max_n = self.params["max_n"]

        cols = [i for i in X.columns if i not in [COL_ID, COL_DATE, COL_WEEK, COL_TARGET]]
        scores = {}
        for col in cols:
            score, ginis = gs_metric(y.values, X[col].values, X["WEEK_NUM"].values, verbose=False, penalty=False)
            scores[col] = score

        correlation_matrix = X[cols].corr()
        selected_models = []
        sorted_models = sorted(scores, key=scores.get, reverse=True)

        i = 0
        for model in sorted_models:
            if all(correlation_matrix[model][selected] < corr_threshold for selected in selected_models):
                selected_models.append(model)
                ensemble_preds = X[selected_models].mean(axis=1).values
                score, ginis = gs_metric(y.values, ensemble_preds, X[COL_WEEK].values, verbose=False, penalty=False)
                print(f"Model {i}. Adding {model}: individual score {scores[model]:.4f}, ensemble score {score:.4f}")
                i += 1
                if len(selected_models) >= max_n:
                    break
            else:
                print(f"Scipping {model}: correlation is too high")

        return selected_models

    def find_best_ensemble_forward(
            self, 
            X: pd.DataFrame, 
            y: pd.Series,
            max_n: int = 10,
            weights: bool = False
        ):
        """
        Find the best combination of models.
        Iteratively adds models to the ensemble based on total score.

        Arguments:
            X: Features to fit on.
            y: Target variable.
            max_n: Maximum number of models to add.
            weights: If True, will use the weights for the ensemble.
        Returns:
            list: List of selected models.
        """
        if "max_n" in self.params:
            max_n = self.params["max_n"]
        if "weights" in self.params:
            weights= self.params["weights"]
        cols = [i for i in X.columns if i not in [COL_ID, COL_DATE, COL_WEEK, COL_TARGET]]
        cols_best = []
        score_best = 0.0
        while True:
            cols_left = [i for i in cols if i not in cols_best]
            col_best = None
            for col in tqdm(cols if weights else cols_left):
                cols_tmp = cols_best.copy()
                cols_tmp.append(col)
                preds = X[cols_tmp].mean(axis=1).values
                score, ginis = gs_metric(y.values, preds, X["WEEK_NUM"].values, verbose=False, penalty=False)
                if score > score_best:
                    print(f"{col}: {score:.4f}, {ginis['slope']:.4f}")
                    score_best = score
                    col_best = col

            if col_best is not None:
                cols_best.append(col_best)
            else:
                print(cols_best)
                break

            if len(cols_best) >= max_n:
                break

        return cols_best

    def find_best_ensemble_corr(self, X: pd.DataFrame, y, score_threshold: float = 0.65, max_n: int = 10):
        """
        Find the best combination of models.
        Iteratively adds models to the ensemble based on correlation with a threshold for score.

        Arguments:
            X: Features to fit on.
            y: Target variable.
            score_threshold: Score threshold.
            max_n: Maximum number of models to add.
        Returns:
            list: List of selected models.
        """
        if "score_threshold" in self.params:
            score_threshold = self.params["score_threshold"]
        if "max_n" in self.params:
            max_n = self.params["max_n"]
        cols = [i for i in X.columns if i not in [COL_ID, COL_DATE, COL_WEEK, COL_TARGET]]
        scores = {}
        for col in cols:
            score, ginis = gs_metric(y.values, X[col].values, X[COL_WEEK].values, verbose=False, penalty=False)
            if score >= score_threshold:
                scores[col] = score

        cols = list(scores.keys())
        sorted_models = sorted(scores, key=scores.get, reverse=True)
        selected_models = [sorted_models[0]]
        print(f"Model {0}. Starting ensemble with {selected_models[0]}: score {scores[selected_models[0]]:.4f}")

        correlation_matrix = X[cols].corr()

        i = 1
        while len(selected_models) < max_n:
            average_correlations = correlation_matrix[selected_models].mean(axis=1).drop(labels=selected_models, errors='ignore')
            next_model = average_correlations.idxmin()
            selected_models.append(next_model)

            ensemble_preds = X[selected_models].mean(axis=1).values
            score, ginis = gs_metric(y.values, ensemble_preds, X[COL_WEEK].values, verbose=False, penalty=False)
            print(f"Model {i}. Adding {next_model}: score {score:.4f}")

            # Break if no more models can be evaluated
            if average_correlations.empty:
                break

            i += 1

        return selected_models