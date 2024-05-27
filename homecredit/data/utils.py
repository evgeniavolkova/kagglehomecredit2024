"""Data processing utilities."""

import gc
import os
import pickle
import shutil
from glob import glob

import numpy as np
import pandas as pd
import polars as pl
from tqdm.auto import tqdm

from ..config import COL_DATE, COL_ID, COL_WEEK, PATH_DATA, PATH_FEATURES

COLS_BOOL = [
    "isbid",
    "isdebit",
]

def gen_file_path(file_name: str, mode: str) -> str:
    """
    Generate data file path.

    Arguments:
        file_name: Name of the file.
        mode: train or test.
    Returns:
        path: Path to the file.
    """
    path = f"{PATH_DATA}/parquet_files/{mode}/{mode}_{file_name}.parquet"
    return path

def set_dtypes(schema: dict) -> dict:
    """
    Set dtypes for the columns in the DataFrame.

    Arguments:
        schema: A dictionary with column names as keys and dtypes as values.
    Returns:
        schema: A dictionary with column names as keys and changed dtypes as values.
    """
    for col, dtype in schema.items():
        if col in [COL_ID,COL_WEEK, "num_group1", "num_group2"]:
            schema[col] = pl.Int64
        elif col in [COL_DATE]:
            schema[col] = pl.Date
        elif col[-1] in ("P", "A"):
            schema[col] = pl.Float64
        elif col[-1] in ("M",):
            schema[col] = pl.String
        elif col[-1] in ("D",):
            schema[col] = pl.Date
        elif col[-1] in ("T", "L"):
            schema[col] = dtype
            for col1 in COLS_BOOL:
                if col1 in col:
                    schema[col] = pl.Int8

    return schema

def write_data_props(dfs_props: dict, version: str = "2") -> None:
    """
    Write the properties of the DataFrames to a pickle file.

    Add information about:
        - paths to files with train and test data
        - schemas
        - categorical columns
        - changed column names

    Arguments:
        dfs_props: A dictionary containing the properties of the DataFrames.
        version: Version of the features configuration.
    """
    # Read features information
    features_df = pd.read_csv(
        os.path.join(PATH_FEATURES, f"features_{version}.csv")
    )
    features_df.fillna("", inplace=True)
    features_df.set_index("feature", inplace=True)

    dates_df = features_df[features_df["date_col"]==1]

    # Loop over data groups and write properties
    for name, props in tqdm(dfs_props.items()):
        features_sg_df = features_df[features_df["source_group"]==name]

        # Dictionary with unprocessed file names (patterns, example: 'applprev_1_*')
        file_names = props["paths"]
        # Initialize variables
        structure_dict = {
            i: {
                "paths": {"train": {}, "test": []}, # Actual paths to data files
                "schema": {}, # Schema of the raw file
                "columns_map": {} # Mapping of columns names
            }
            for i in file_names
        }
        # Column mappings are required to merge data from different sources.
        # If we have more than one source, we rename columns
        # from different sources to corresponding columns from
        # one of the sources ('a'). New names are set in features.csv
        columns = [] # List with columns
        cols_cat = {}
        columns_dtypes = {} # List with dtypes for all columns

        for file_name in file_names: # Loop over each file in the group
            for mode in ["train", "test"]: # Loop over modes
                # Get actual paths to files
                paths_i = glob(gen_file_path(file_name, mode))
                paths_i = sort_paths(paths_i)

                # Save schema from the first train file
                if mode == "train":
                    df = pl.read_parquet(paths_i[0])
                    schema = set_dtypes(df.schema)
                    structure_dict[file_name]["schema"] = schema

            # Save columns, dtypes and mappings
            cols_to_incl = [] # List with columns
            cols_cat_i = {} # Dict with categorical columns and their values
            cols_dtypes = {} # Dict with column dtypes
            mapping = {col: None for col in df.columns} # Column mapping
            for col in df.columns:
                # Create mapping
                if col in features_sg_df.index.values:
                    col_new = features_sg_df.loc[col]["new_name"]
                    mapping[col] = col_new if col_new else col
                    if features_df.loc[col]["agg"] == "dummy":
                        col_vals = [i for i in df[col].unique() if (i is not None) & (i != "")]
                        cols_cat_i.update({col: col_vals})
                else:
                    mapping[col] = col

                # Add column name
                cols_to_incl.append(mapping[col])

                # Add column dtype
                cols_dtypes.update({col: schema[col]})

            # Update columns and dtypes
            columns_dtypes.update(cols_dtypes)
            columns.extend(cols_to_incl)
            cols_cat.update(cols_cat_i)

            # Add mapping
            structure_dict[file_name]["columns_map"] = mapping

            del df
            gc.collect()
        columns = np.unique(columns)

        # Update props
        props["structure"] = structure_dict
        props["columns"] = columns
        props["columns_cat"] = cols_cat
        props["columns_dtypes"] = columns_dtypes
        props["columns_date_index"] = dates_df[dates_df["source_group"]==name].index.to_list()

    # Save
    with open(os.path.join(PATH_FEATURES, f"dfs_props_{version}.pkl"), 'wb') as handle:
        pickle.dump(dfs_props, handle)

def sort_paths(paths: list) -> list:
    """
    Sort paths to homecredit files.

    Arguments:
        paths: list of paths. Each path should be in the format {name}_{n}.
    Returns:
        sorted_paths: sorted list of paths.
    """
    if len(paths) == 1:
        return paths
    else:
        paths_nums = [int(path.split("_")[-1].split(".")[0]) for path in paths]
        sorted_paths = [path for _, path in sorted(zip(paths_nums, paths))]
        return sorted_paths

def create_folder(path: str, rm: bool = False) -> None:
    """
    Create a folder with os.makedirs.

    Arguments:
        path: Path to the folder.
        rm: Whether to remove path if it exists.
    """
    if rm:
        if os.path.exists(path):
            shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def get_features_df(dfs_props, version: str = "2") -> pd.DataFrame:
    """
    Loads and processes the features configuration from a CSV file.

    Arguments:
        dfs_props: A dictionary containing the properties of the DataFrames.
        version: Version of the features configuration.
    Returns:
        features_df: A DataFrame with feature information.
    """
    # Read csv with feature information
    features_df = pd.read_csv(
        os.path.join(PATH_FEATURES, f"features_{version}.csv"),
        index_col=0,
    )
    features_df = features_df[features_df["source_group"].isin(dfs_props.keys())]
    features_df.fillna("", inplace=True)

    return features_df

def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce memory usage of a DataFrame.

    Arguments:
        df: DataFrame to reduce memory usage.
    Returns:
        df: DataFrame with reduced memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    cols = df.select_dtypes("number").columns

    for col in cols:
        col_type = df[col].dtype
        c_min = df[col].min()
        c_max = df[col].max()
        if str(col_type)[:3] == 'int':
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                df[col] = df[col].astype(np.int64)  
        else:
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                df[col] = df[col].astype(np.float16)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df