"""Models analysis: feature selection, feature importances, etc."""

import random

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from tqdm.auto import tqdm

from .config import COL_DATE, COL_ID, COL_TARGET, COL_WEEK, base_path
from .models.tree import LGBM
from .pipeline import FullPipeline


def select_features(
        features: list, 
        score_func: callable, 
        features_all: list | None = None, 
        method: str = "forward", 
        fast: bool = False, 
        superfast: bool = False, 
        shuffle: bool = False, 
        chunk_size: int = 1, 
        threshold: float = 0.0
    ):
    """
    Backward/forward feature selection.

    Arguments:
        features: list of features to start with
        score_func: function to calculate the score
        features_all: list of all features to select from
        method: "forward" or "backward"
        fast: stop after the first feature is added on each iteration
        superfast: don't try features that were already tried
        shuffle: shuffle the features on each iteration
        chunk_size: number of features to add/remove at once
        threshold: score threshold to add/remove a feature
    Returns:
        res_df: DataFrame with scores
    """
    
    def chunks(lst, chunk_size):
        for i in range(0, len(lst), chunk_size):
            yield lst[i:i + chunk_size]

    features_incl = features.copy()
    features_selected = []
    if method=="forward":
        features_left = [i for i in features_all if i not in features_incl]
    else:
        features_left = features_all.copy()

    res_df = pd.DataFrame(index=features_left)

    if len(features_incl) > 0:
        scores = score_func(features=features_incl)
        score_best = scores["score"]
        score_mean_best = scores["mean"]
        slope_best = scores["slope"]
        std_best = scores["std"]
        print(f"Initial score: {score_best:.4f}, mean {score_mean_best:.4f}, std {std_best:.4f}, slope {slope_best:.4f}")
    else:
        score_best = 0.0
        score_mean_best = 0.0
        slope_best = 0.0
        std_best = 0.0
        print(f"No initial features. Initial score: {score_best:.4f}")

    cnt = 0
    while True:
        res_df[cnt] = np.nan
        feature_selected = None
        if shuffle:
            random.shuffle(features_left)
        pbar = tqdm(chunks(features_left, chunk_size), total=len(features_left)//chunk_size)
        cnt1 = 0
        for f in pbar:
            features_tmp = features_incl.copy()
            if method == "forward":
                features_tmp.extend(f)
            else:
                for i in f:
                    features_tmp.remove(i)
            scores = score_func(features=features_tmp)
            score = scores["score"]
            score_mean = scores["mean"]
            slope = scores["slope"]
            std = scores["std"]

            res_df.loc[f, cnt] = score

            cnt1 += len(f)

            if score - score_best > threshold:
                score_best = score
                score_mean_best = score_mean
                slope_best = slope
                feature_selected = f
                print(f"score {score_best:.4f}, mean {score_mean:.4f}, std {std:.4f}, slope {slope:.4f}: {feature_selected}")
                if fast:
                    break

        if feature_selected is not None:
            if method=="forward":
                features_incl.extend(feature_selected)
            else:
                for i in feature_selected:
                    features_incl.remove(i)
            features_selected.extend(feature_selected)
            if superfast:
                features_left = features_left[cnt1:]
            else:
                for i in feature_selected:
                    features_left.remove(i)
            print(f"Score {score_best:.4f}. {feature_selected} was selected.")
            print(features_selected)
        else:
            print(f"Score {score_best:.4f}. No good features left. Stopping.")
            break

        cnt += 1

    return res_df


def get_null_imp(df: pd.DataFrame, features: list, model_name: str, params: dict, n = 10) -> pd.DataFrame:
    """
    Get null importances for the features.

    Arguments:
        df: pandas DataFrame with the data
        features: list of features
        model_name: name of the model
        params: parameters for the model
        n: number of null importances to calculate
    Returns:
        imp_df: DataFrame with importances
    """
    params_null = params.copy()
    params_null["n_estimators"] = 300
    model = LGBM(params_null, early_stopping=False, test_size=0.01, shuffle=True)
    imps = []
    for i in range(n):
        df["target_shuffled"] = df["target"].sample(frac=1, random_state=i).values
        name = f"{model_name}_null_{i}"
        pipeline = FullPipeline(
            Pipeline(steps=[
                ('classifier', model),
            ]),
            run_name="full",
            name=name,
            load_model=False,
            features=features,
            target_col="target_shuffled",
        )
        pipeline.fit(df, verbose=True)
        imps.append(pipeline.model["classifier"].get_feature_importances()[["imp"]].add_suffix(f"_{i}"))

    imp_df = pd.concat(imps, axis=1)
    return imp_df

def select_features_null_imp(version: str, df: pd.DataFrame, n_splits: int = 2) -> None:
    """
    Select features based on null importances.

    Arguments:
        version: version of the model
        df: pandas DataFrame with the data
        n_splits: number of splits for the null importances
    Returns:
        features_selected: list of selected features
    """
    features_all = np.array([i for i in df.columns if i not in [COL_ID, COL_DATE, COL_WEEK, COL_TARGET]])
    random.shuffle(features_all)
    n = n_splits
    part_size = len(features_all) // n
    remainder = len(features_all) % n
    indices = [(i + 1) * part_size + (1 if i < remainder else 0) for i in range(n)]
    feature_sets = [part.tolist() for part in np.array_split(features_all, indices[:-1])]

    model_name = f"lgb_version_{version}_nullimp"

    params_lgb = {
        'boosting_type': 'gbdt',
        'colsample_bynode': 0.8,
        'colsample_bytree': 0.8,
        'extra_trees': True,
        'learning_rate': 0.1,
        'max_depth': 10,
        'metric': 'auc',
        'n_estimators': 4000,
        'num_leaves': 64,
        'objective': 'binary',
        'random_state': 42,
        'reg_alpha': 10,
        'reg_lambda': 10,
        "device": "gpu",
        'verbose': -1,
        "max_bin": 150,
    }
    imps = []
    imps_null = []
    model = LGBM(params_lgb, early_stopping_rounds=50, test_size=0.01, shuffle=True)
    for i, features in tqdm(enumerate(feature_sets), total=len(feature_sets)):
        name = f"{model_name}_full_{i}"
        print(f"Number of features {len(features)}")
        pipeline = FullPipeline(
            Pipeline(steps=[
                ('classifier', model),
            ]),
            run_name="full",
            name=name,
            load_model=False,
            features=features
        )
        df_sample = df.sample(frac=0.5, random_state=i)
        pipeline.fit(df_sample, verbose=True)
        imp_i_df = pipeline.model["classifier"].get_feature_importances()[["imp"]]
        imps.append(imp_i_df)

        imp_null_i_df = get_null_imp(df_sample, features, f"{model_name}_full_{i}", params_lgb, n = 10)
        imps_null.append(imp_null_i_df)

    imp_df = pd.concat(imps, axis=0)
    imp_null_df = pd.concat(imps_null, axis=0)

    imp_df["imp_null_75"] = imp_null_df.quantile(0.75, axis=1)
    imp_df["imp_final"] = imp_df["imp"] / (imp_df["imp_null_75"]+0.01)
    imp_df.sort_values("imp_final", ascending=False, inplace=True)
    features_selected = imp_df[(imp_df["imp_final"] > 1)&(imp_df["imp"]>1)].index.to_list()

    imp_df.to_csv(f"{base_path}/analysis/imp_null_{version}.csv")

    return features_selected

def select_features_adv(version: str, df: pd.DataFrame, features: list) -> list:
    """
    Select features based on adversarial validation.

    Select features that are not predictive of time period (covid).

    Arguments:
        version: version of the model
        df: pandas DataFrame with the data
        n_splits: number of splits for the null importances
    Returns:
        features_selected: list of selected features
    """
    df["covid"] = 0
    df.loc[df["WEEK_NUM"]>=65, "covid"] = 1

    df_train, df_valid = train_test_split(df, random_state=42)

    params_lgb = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        "n_estimators": 20,
        'random_state': 42,
        "device": "gpu",
        'verbose': -1,
    }
    model = LGBM(params_lgb, early_stopping_rounds=200, test_size=0.01, shuffle=True)

    features = features
    print(f"Number of features: {len(features)}")

    res_df = pd.DataFrame(index=features, columns=["score"])

    for f in tqdm(features):
        pipeline = FullPipeline(
            Pipeline(steps=[
                ('classifier', model),
            ]),
            run_name="adv",
            name="test",
            load_model=False,
            features=[f],
            target_col="covid"
        )
        try:
            pipeline.fit(df_train)
        except:
            continue
        preds = pipeline.predict_proba(df_valid)
        score = roc_auc_score(df_valid["covid"], preds)
        res_df.loc[f, "score"] = score

    res_df.to_csv(f"{base_path}/analysis/adv_val_features_{version}.csv")
    features_stable = res_df.loc[res_df["score"]<0.65].index.to_list()
    return features_stable