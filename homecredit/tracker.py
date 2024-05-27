"""Evaluation and logging functions."""

import numpy as np
import pandas as pd
import wandb

from .config import COL_TARGET
from .metrics import gs_metric
from .pipeline import FullPipeline, PipelineCV


class WandbTracker:
    """ Wandb tracker custom class."""

    def __init__(self, run_name: str, params: dict, category: str, comment: str) -> None:
        """
        Initialize the Wandb tracker.

        Arguments:
            run_name: Name of the run.
            params: Dictionary with parameters for the run.
            category: Category of the run.
            comment: Comment for the run.
        """
        self.run_name = run_name
        self.params = params
        self.category = category
        self.comment = comment
        self.api = wandb.Api()

    def init_run(self, features: list) -> None:
        """ 
        Initialize the run.
        
        Arguments:
            features: List of features used in the model.
        """

        config = self.params.copy()
        config.update({
            "model": "lgb",
            "category": self.category,
            "comment": self.comment,
            "n_features": len(features)
        })
        wandb.init(
            project="kaggle_home_credit",
            name=self.run_name,
            config=config
        )

    def save_features(self, features: list) -> None:
        """
        Save the list of features as an artifact.

        Arguments:
            features: List of features used in the model.
        """
        feature_file_path = "features.txt"
        with open(feature_file_path, "w") as file:
            for feature in features:
                file.write(f"{feature}\n")
        artifact = wandb.Artifact(name=f"{self.run_name}-feature-list", type="dataset")
        artifact.add_file(feature_file_path)
        wandb.log_artifact(artifact)

    def alert(self, text: str) -> None:
        """
        Send an alert to the user.

        Arguments:
            text: Text of the alert.
        """
        wandb.alert(
            title=f'Run {self.run_name} finished.',
            text=text,
            level=wandb.AlertLevel.INFO
        )

    def log_metrics(self, metrics: dict) -> None:
        """
        Log metrics to the run.

        Arguments:
            metrics: Dictionary with metrics to log.
        """
        wandb.log(metrics)

    def update_summary(self, run_id: str, summary_params: dict) -> None:
        """
        Update the summary of the run.

        Arguments:
            run_id: ID of the run.
            summary_params: Dictionary with the summary parameters to update.
        """
        run = self.api.run(f"eivolkova3/kaggle_home_credit/{run_id}")
        for key, val in summary_params.items():
            run.summary[key] = val
        run.summary.update()

    def update_settings(self, run_id: str, settings_params: dict) -> None:
        """
        Update the settings of the run.

        Arguments:
            run_id: ID of the run.
            settings_params: Dictionary with the settings parameters to update.
        """
        run = self.api.run(f"eivolkova3/kaggle_home_credit/{run_id}")
        for key, val in settings_params.items():
            run.settings[key] = val
        run.update()

    def finish(self) -> None:
        """ Finish the run."""
        wandb.finish()


def evaluate_and_log(
        df: pd.DataFrame, 
        params: dict, 
        pipeline: FullPipeline,
        n_splits: int = 3,
        features: list | None = None,
        model_name: str = "model", 
        comment: str = "",
        track: bool = False, 
        verbose: bool = False,
        ts: bool = False, 
        category: str | None = None
    ):
    """
    Evaluate the pipeline and log the results to Wandb.

    Arguments:
        df: pandas DataFrame with the data
        params: dictionary with parameters to log
        pipeline: FullPipeline object (not fitted)
        n_splits: number of splits for the cross-validation
        features: list of features
        model_name: name of the model
        comment: comment to log
        track: whether to track the run
        verbose: whether to print the results
        ts: whether to use time series CV
        category: category to log
    Returns:
        cv: PipelineCV object
        preds_df: DataFrame with predictions
    """
    if category is None:
        category = "model_ver12" if ts else "model_ver13"
    
    params = dict(params)
    params["n_splits"] = n_splits
    if track:
        wandb_tracker = WandbTracker(
            model_name,
            params,
            category=category,
            comment=comment
        )
        wandb_tracker.init_run(features)
        wandb_tracker.save_features(features)

    cv = PipelineCV(pipeline, weights=False, n_splits=n_splits)
    preds_df = cv.fit(df, verbose=verbose)

    score, ginis_dict = gs_metric(preds_df[COL_TARGET].values, preds_df["preds"].values, preds_df["WEEK_NUM"].values, penalty=False, verbose=verbose)

    if track:
        metrics = {
            "cv": score,
            "cv_mean": np.mean(ginis_dict["ginis"]),
            "cv_slope": ginis_dict["slope"],
            "cv_std": ginis_dict["std"]
        }
        wandb_tracker.log_metrics(metrics)

        alert_text = (f"CV score: {score:.3f}. "
                      f"CV mean: {np.mean(ginis_dict['ginis']):.3f}. "
                      f"CV slope: {ginis_dict['slope']:.3f}. "
                      f"CV std: {ginis_dict['std']:.3f}.")

        wandb_tracker.alert(alert_text)
        wandb_tracker.finish()

    return cv, preds_df
