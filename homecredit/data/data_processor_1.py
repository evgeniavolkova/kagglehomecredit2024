"""Data processing: version 1."""

import ast
import os
import pickle
from glob import glob
from typing import Iterator

import numpy as np
import pandas as pd
import polars as pl
from tqdm.auto import tqdm

from ..config import (COL_DATE, COL_ID, COL_TARGET, COL_WEEK, KAGGLE,
                      PATH_DATA_PROC, PATH_FEATURES)
from . import utils_old


class PolarsExprTransformer(ast.NodeTransformer):
    """
    Safely evaluate a Polars expression from a string.
    """
    def visit_Constant(self, node):
        return node.value

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        
        if isinstance(node.op, ast.Add):
            return left + right
        elif isinstance(node.op, ast.Sub):
            return left - right
        elif isinstance(node.op, ast.Mult):
            return left * right
        elif isinstance(node.op, ast.Div):
            return left / right
        elif isinstance(node.op, ast.Pow):
            return left.pow(right)
        elif isinstance(node.op, ast.Gt):
            return left.gt(right)
        elif isinstance(node.op, ast.Lt):
            return left.lt(right)
        elif isinstance(node.op, ast.GtE):
            return left.ge(right)
        elif isinstance(node.op, ast.LtE):
            return left.le(right)
        else:
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
        
    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == 'col':
                col_name = node.args[0].s
                return pl.col(col_name)
            elif node.func.attr == 'alias':
                expr = self.visit(node.func.value)
                alias_name = node.args[0].s
                return expr.alias(alias_name)
            elif node.func.attr in ['coalesce']:
                args = [self.visit(arg) for arg in node.args]
                return pl.coalesce(*args)
            elif node.func.attr in ['date']:
                args = [self.visit(arg) for arg in node.args]
                return pl.date(*args)
            elif node.func.attr == 'fill_null':
                expr = self.visit(node.func.value)
                fill_value = self.visit(node.args[0])
                return expr.fill_null(fill_value)
            elif node.func.attr == 'mean':
                expr = self.visit(node.func.value)
                return expr.mean()
            elif node.func.attr == 'diff':
                expr = self.visit(node.func.value)
                return expr.diff()
            elif node.func.attr == 'gt':
                expr = self.visit(node.func.value)
                fill_value = self.visit(node.args[0])
                return expr.gt(fill_value)
        else:
            raise ValueError(f"Unsupported function call: {ast.dump(node)}")

    def visit_Name(self, node):
        return pl.col(node.id)

class Utils:
    """
    Class for utility functions to process data.
    """
    @staticmethod
    def cast_dtypes(df, schema):
        expr_ls = []
        for col in df.columns:
            expr_ls.append(pl.col(col).cast(schema[col]))
        df = df.with_columns(expr_ls)
        return df
    
    @staticmethod
    def handle_cat(df: pl.DataFrame) -> pl.DataFrame:
        """
        Replace missing values in categorical columns with None.

        Arguments:
            df: polars.DataFrame with data.
        Returns:
            df: polars.DataFrame with missing values replaced.
        """
        df = df.with_columns([
            pl.when(pl.col(pl.Utf8) == "a55475b1")
            .then(None)
            .otherwise(pl.col(pl.Utf8))
            .keep_name()
        ])
        return df

    @staticmethod
    def agg_process(col: str, oper: str) -> pl.Expr:
        """
        Create an aggregation expression for a column.
        Args:
            col: str, name of the column to aggregate
            oper: str, name of the operation to perform
        Returns:
            expr: polars.Expr, aggregation expression
        """
        operations: dict[str, callable[[], pl.Expr]] = {
            "min": lambda: pl.col(col).min(),
            "max": lambda: pl.col(col).max(),
            "sum": lambda: pl.col(col).sum(),
            "mean": lambda: pl.col(col).mean(),
            "mode": lambda: pl.col(col).drop_nulls().mode().first(),
            "std": lambda: pl.col(col).std(),
            "count": lambda: pl.col(col).count(),
            "last": lambda: pl.col(col).drop_nulls().last(),
            "first": lambda: pl.col(col).drop_nulls().first(),
            "q05": lambda: pl.col(col).quantile(0.05),
            "q95": lambda: pl.col(col).quantile(0.95),
            "median": lambda: pl.col(col).median(),
            "skew": lambda: pl.col(col).skew(),
            "kurt": lambda: pl.col(col).kurtosis(),
            "range": lambda: pl.col(col).max() - pl.col(col).min(),
            "rangefirstmax": lambda: pl.col(col).drop_nulls().first() - pl.col(col).max(),
            "rangefirstlast": lambda: pl.col(col).drop_nulls().first() - pl.col(col).drop_nulls().last(),
            "rangelastfirst": lambda: pl.col(col).drop_nulls().last() - pl.col(col).drop_nulls().first(), 
            "share": lambda: pl.col(col).sum() / pl.col(col).count(),
            "nunique": lambda: pl.col(col).n_unique(),
            "nuniquetotal": lambda: pl.col(col).n_unique() / pl.col(col).count(),
            "sharenonzero": lambda: pl.col(col).filter(pl.col(col) != 0).count() / pl.col(col).count(),
            "meannonzero": lambda: pl.col(col).filter(pl.col(col) != 0).mean(),
        }

        if oper in operations:
            return operations[oper]()
        else:
            raise ValueError(f"Operation '{oper}' is not supported.")

    @staticmethod
    def prepare_formula(row: pd.Series, col: str, i: int | None = None) -> list[pl.Expr]:
        """
        Prepares a Polars expression formula based on the input parameters and the specified feature transformation stage.

        Arguments:
            row (dict): A dictionary with feature information.
            col (str): Feature name.
            i (Optional[int]): Formula index.
            stage (int): The transformation stage of the feature (0 - raw data, 1 - aggregated data).
        Returns:
            List[pl.Expr]: A list containing Polars expressions.
        """
        key = f"formula1{i if i is not None else ''}"
        formula = row[key]

        placeholders = {"x": f"pl.col('{col}')"}
        placeholders.update({
            f"x1{i}": f"pl.col('{row[f'x1{i}']}')" for i in range(1, 5)
        })

        formula = formula.format(**placeholders)
        expr = Utils.parse_polars_formula(formula)
        return expr

    @staticmethod
    def parse_polars_formula(formula) -> pl.Expr:
        """
        Parse polars formula from a string.

        Arguments:
            formula (str): A string containing the formula.
        Returns:
            pl.Expr: A Polars expression.
        """
        expr_ast = ast.parse(formula, mode='eval').body
        transformer = PolarsExprTransformer()
        polars_expr = transformer.visit(expr_ast)
        return polars_expr

    @staticmethod
    def change_dtypes(df: pl.DataFrame, dtype_in: list[pl.DataType], dtype_out: pl.DataType) -> pl.DataFrame:
        df = df.with_columns(
            [
                pl.col(col).cast(dtype_out)
                for col
                in df.select(pl.selectors.by_dtype(*dtype_in)).columns
            ]
        )
        return df

    @staticmethod
    def downcast(df):
        df = df.with_columns(pl.col(pl.NUMERIC_DTYPES).shrink_dtype())
        df = df.with_columns(pl.col(COL_ID).cast(pl.Int64))
        return df

    @staticmethod
    def filter_cols(df: pd.DataFrame) -> pd.DataFrame:
        cols_to_drop = []
        for col in df.columns:
            if col not in [COL_TARGET, COL_ID, COL_DATE, COL_WEEK]:
                isnull = df[col].isna().mean()
                if isnull > 0.7:
                    cols_to_drop.append(col)
        
        cat_cols = df.select_dtypes("category")
        for col in cat_cols:
            freq = df[col].nunique()
            if (freq == 1) | (freq > 200):
                cols_to_drop.append(col)
        df = df.drop(columns=cols_to_drop)
        return df
    
    @staticmethod
    def match_cols(dfs):
        columns = {}
        for df in dfs:
            columns.update({col: df[col].dtype for col in df.columns})

        for i in range(len(dfs)):
            for col, dtype in columns.items():
                if col not in dfs[i].columns:
                    dfs[i] = dfs[i].with_columns(pl.lit(None).cast(dtype).alias(col))
            dfs[i] = dfs[i].select(list(columns.keys()))

        return dfs

class BatchReader:
    """
    A utility class designed to read large datasets in batches efficiently.
    It supports reading data in train or test mode and ensures continuity
    of case IDs across batches.
    """
    def __init__(self, df_props: dict, test: bool, batch_size: int = 10**6):
        """
        Initializes the BatchReader with dataset properties, mode, and batch size.

        Parameters:
            df_props (dict): Properties of the dataframe to read, must include 'path' and 'schema'.
            test (bool): Flag indicating whether to read data in test mode. False means train mode.
            batch_size (int): The number of rows per batch. Should be set according to memory limits.
        """
        self.df_props = df_props
        self.mode = "train" if not test else "test"
        self.batch_size = batch_size

        self.df_last_case_id = None
        self.rows_proc = None
        self.csv_len = None

    def _match_cols(self, df: pl.DataFrame, file_name: str) -> pl.DataFrame:
        """

        """
        cols = df.columns

        # Rename columns
        mapping = self.df_props["structure"][file_name]["columns_map"]
        df = df.rename({key: val for key, val in mapping.items() if key in df.columns})

        # Drop redundant cols
        cols_to_drop = [col for col in cols if col not in self.df_props["columns"]]
        df = df.drop(cols_to_drop)

        # Add missing cols
        cols = self.df_props["columns"]
        for col in cols:
            if col not in df.columns:
                df = df.with_columns(pl.lit(None).cast(self.df_props["columns_dtypes"][col]).alias(col))

        # Reorder
        df = df.select(self.df_props["columns"])

        return df

    def _handle_last_case_id(self, df: pl.DataFrame) -> pl.DataFrame:
        file_end = self.rows_proc == self.csv_len
        last_case_exists = self.df_last_case_id is not None

        if (len(df) == 0) & ((not file_end) or (file_end & (not last_case_exists))):
            return df

        # Append last case_id from the previous batch
        if last_case_exists:
            df = pl.concat([self.df_last_case_id, df], how="vertical_relaxed")

        # Handle last case_id for continuity with the next batch
        if not file_end:
            last_case_id = df.select(pl.last(COL_ID)).to_series()[0]
            self.df_last_case_id = df.filter(pl.col(COL_ID) == last_case_id)
            df = df.filter(pl.col(COL_ID) != last_case_id)
        else:
            self.df_last_case_id = None

        return df

    def batches(self, case_ids) -> Iterator[pl.DataFrame]:
        """
        Yields data batches from the dataset group, ensuring continuity of case IDs across batches.

        This method reads data in chunks defined by `batch_size`, matches data schema,
        matches columns from different sources, and handles cases where a
        case ID spans multiple batches, ensuring that all data related to a single case ID is processed
        together.

        Returns:
            Iterator[pl.DataFrame]: An iterator over the data batches.
        """
        str_dict = self.df_props["structure"]
        for file_name, val in str_dict.items():
            paths_i = glob(utils_old.gen_file_path(file_name, self.mode))
            paths_i = utils_old.sort_paths(paths_i)
            for path in paths_i:
                df_lazy = pl.scan_parquet(path)
                case_ids_all = df_lazy.select(COL_ID).collect()[COL_ID].to_list()
                n = len(case_ids_all)
                n_batches = int(np.ceil(n/self.batch_size))

                for i in range(n_batches):
                    df = df_lazy.filter(pl.col(COL_ID).is_in(case_ids_all[i*self.batch_size:(i+1)*self.batch_size])).collect()

                    df = Utils.cast_dtypes(df, str_dict[file_name]["schema"])
                    df = self._match_cols(df, file_name)
                    if case_ids is not None:
                        df = df.filter(pl.col(COL_ID).is_in(case_ids))

                    yield df, file_name


class BatchDataHandler:
    """
    Handles batch data processing for a given dataset.
    """
    DEFAULT_BATCH_SIZE = 10**7
    TEST_BATCH_SIZE = 3 * 10**5
    MERGE_COLS =  [COL_TARGET, COL_DATE, COL_WEEK]
    MERGE_COLS_S0 = ["credtype_322L_1_static0", "credamount_770A_1_static0"]

    def __init__(
            self,
            name: str,
            df_props: dict,
            features_df: pd.DataFrame,
            fit: bool,
            test: bool
        ):
        """
        Initializes the BatchDataHandler.

        Arguments:
            name: Name of dataset.
            df_props: Specifications of the dataset.
            features_df: A dictionary containing features and their respective processing information.
            fit: Specifies if the handler should fit submodels or load them.
            test: Specifies if the handler is operating in test mode.
        """
        self.name = name
        self.df_props = df_props
        self.features_df = features_df
        self.fit = fit
        self.test = test

        self.batch_size = self.TEST_BATCH_SIZE if KAGGLE else self.DEFAULT_BATCH_SIZE

    def transform(self, df_init: pl.DataFrame, case_ids: list | None) -> pl.DataFrame:
        """
        Transform the data in batches and concatenate the results.

        Arguments:
            df_init: The initial DataFrame with base information..
            case_ids: A list of case IDs to filter the data.
        Returns:
            pl.DataFrame: The transformed DataFrame.
        """
        batch_reader = BatchReader(self.df_props, self.test, batch_size=self.batch_size)
        dfs = []
        for batch_num, (df, source) in enumerate(batch_reader.batches(case_ids)):
            df = self._transform_data(df, df_init, batch_num, source)
            dfs.append(df)

        dfs = Utils.match_cols(dfs)
        df = pl.concat(dfs, how="vertical_relaxed")
        df = self._post_process_batches(df, df_init)
        return df

    def _transform_data(self, df: pl.DataFrame, df_init: pl.DataFrame, batch_num: int, source: str) -> pl.DataFrame:
        """
        Transform the data in a single batch.

        Filter the data based on case IDs, join it with the initial DataFrame, and perform feature engineering.

        Arguments:
            df: The DataFrame with the batch data.
            df_init: The initial DataFrame with base information.
            batch_num: The number of the batch.
            source: The source of the data.
        Returns:
            pl.DataFrame: The transformed DataFrame.
        """
        merge_cols = [COL_ID] + self.MERGE_COLS
        merge_cols += self.MERGE_COLS_S0 if self.name != "static0" else []
        df = df.join(df_init.select(merge_cols), on=COL_ID, how="inner")

        feature_engineer = FeatureEngineer(self.name, source, self.df_props, self.features_df, batch_num, self.fit, merge_cols)
        df_agg = feature_engineer.transform(df)
        df_agg = df_agg.drop(self.MERGE_COLS + self.MERGE_COLS_S0)
        return df_agg

    def _post_process_batches(self, df: pl.DataFrame, df_init: pl.DataFrame) -> pl.DataFrame:
        """
        Perform post-processing on the concatenated DataFrame.

        Rename columns and join the DataFrame with the initial DataFrame.

        Arguments:
            df: The concatenated DataFrame.
            df_init: The initial DataFrame with base information.
        Returns:
            pl.DataFrame: The post-processed DataFrame.
        """
        df = df.unique(subset=[COL_ID], keep="first")
        df = Utils.downcast(df)
        df = df.rename({col: f"{col}_{self.name}" for col in df.columns if col != COL_ID})
        df = df_init.join(df, on=COL_ID, how="left")
        # df = df.with_columns(pl.col('^.*_dummy$').fill_nan(0))
        return df

class FeatureEngineer:
    """
    A class to handle feature engineering for a given dataset.
    """
    submodel_opers = [
            "sum",
            "max", "min",
            "mean", "median",
            "std", "skew", "kurt",
            "q05", "q95"
    ]

    def __init__(self, name: str, source: str, df_props: dict, features_df: dict, batch_num: int, fit: bool, cols_keep: list) -> None:
        """
        Initializes the FeatureEngineer.

        Arguments:
            name: Name of the dataset.
            run_name: A unique identifier for the current run or operation.
            df_props: Specifications of the dataset.
            features_df: A dictionary containing features and their respective processing information.
            batch_num: The number of the batch.
            fit: Specifies if the handler should fit submodels or load them.
            load: A flag to indicate if the data should be loaded from the disc.
        """
        self.name = name
        self.source = source

        self.df_props = df_props
        self.features_df = features_df

        self.batch_num = batch_num

        self.fit = fit
        self.cols_keep = cols_keep + ["num_group1", "num_group2", "active"]

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Perform feature engineering on the dataset.

        Arguments:
            df: The DataFrame with the batch data.
        Returns:
            pl.DataFrame: The transformed DataFrame.
        """
        # if self.name == "creditbureau1":
        #     df = df.filter(pl.col("classificationofcontr_13M") != "a55475b1")
        # if self.name == "creditbureau1":
        #     df = df.filter(pl.col("classificationofcontr_13M") != "a55475b1")

        # Proc
        df = Utils.change_dtypes(df, [pl.Categorical, pl.Boolean], pl.String)
        df = Utils.handle_cat(df)

        features = [i for i in df.columns if i not in self.cols_keep]
        # df = self._scale_amount(df)
        df = self._process_formulas_all(df, features)
        df = df.drop(features)
        df = self._handle_dates(df)
        # Perform transformations

        depth = int(self.name[-1])
        if depth > 0:
            dfs = []

            # Handle person1
            if self.name == "person1":
                dfi, df = self._handle_person1(df)
                dfs.append(dfi)

            if self.name == "applprev1":
                dfi = self._handle_applprev1(df)
                dfs.append(dfi)

            if "creditbureau" in self.name:
                df = self._handle_contracts_cb(df)

            # Aggregate data
            dfi = self._process_aggregations_all(df)
            dfs.append(dfi)

            df = dfs[0]
            for df_i in dfs[1:]:
                df = df.join(df_i, on=COL_ID, how="outer_coalesce")

        return df

    def _handle_dates(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Create date features from date columns.
        Args:
            df: polars.DataFrame with data.
        Returns:
            df: polars.DataFrame with date features.
        """
        expr_ls = []
        for col in df.columns:
            if df[col].dtype == pl.Date:
                expr_ls.append((pl.col(col) - pl.col(COL_DATE)).dt.total_days().alias(col))
        df = df.with_columns(expr_ls)
        df = df.drop(COL_DATE)
        return df

    def _scale_amount(self, df: pl.DataFrame) -> pl.DataFrame:
        inc_col = "maininc_215A_1_static0" if self.name != "static0" else "maininc_215A"
        inc_other_col = "lastotherinc_902A_1_static0" if self.name != "static0" else "lastotherinc_902A"
        cols_amount = [i for i in df.columns if i[-1] == "A"]
        cols_amount = [i for i in cols_amount if i not in [inc_col]]

        df = df.with_columns(
            pl.col(cols_amount) / (pl.col(inc_col).fill_null(50000)+pl.col(inc_other_col).fill_null(0)+1)
        )
        return df 
    
    def _handle_contracts_cb(self, df: pl.DataFrame) -> pl.DataFrame:
        # Identifying columns based on their roles
        active_cols = []
        closed_cols = []
        common_cols = [COL_ID]
        for col_new in df.columns:
            col = "_".join(col_new.split("_")[:-1])
            if col not in self.features_df.index:
                continue
            if "active" in self.features_df.loc[col, "active"]:
                active_cols.append(col_new)
            if "closed" in self.features_df.loc[col, "active"]:
                closed_cols.append(col_new)
            if "common" in self.features_df.loc[col, "active"]:
                common_cols.append(col_new)

        # Creating a mapping for new column names (active and closed columns)
        cols_map = {}
        for col in active_cols + closed_cols:
            new_col_name = "_".join(col.split("_")[:-2]) + "_all_" + col.split("_")[-1]
            if new_col_name not in cols_map:
                cols_map[new_col_name] = []
            cols_map[new_col_name].append(col)

        map_active = {}
        map_closed = {}
        for new_col, cols in cols_map.items():
            if len(cols) < 2:
                continue
            for col in cols:
                if col in active_cols:
                    map_active[col] = new_col
                if col in closed_cols:
                    map_closed[col] = new_col

        # Prepare two DataFrames: one for active columns and one for closed columns
        active_df = df.select(common_cols + active_cols).rename(map_active)
        closed_df = df.select(common_cols + closed_cols).rename(map_closed)

        # Add an "active" column to indicate the status
        active_df = active_df.with_columns(pl.lit(True).alias("active"))
        closed_df = closed_df.with_columns(pl.lit(False).alias("active"))

        # Add missing columns filled with NaN to each DataFrame
        active_df, closed_df = Utils.match_cols([active_df, closed_df])

        # Concatenate active and closed DataFrames
        df_new = pl.concat([active_df, closed_df])

        return df_new
    
    def _handle_person1(self, df: pl.DataFrame) -> pl.DataFrame:
        df_persons = df.filter(pl.col("num_group1")==0)
        df_persons = df_persons.drop(["num_group1"])
        features_drop = self.features_df[self.features_df["include"]==0].index.to_list()
        features_drop = [f"{item}_{i}" for item in features_drop for i in range(1, 4)]
        df_persons = df_persons.drop(features_drop)
        df = df.filter(pl.col("num_group1")!=0)
        return df_persons, df
    
    def _handle_applprev1(self, df: pl.DataFrame) -> pl.DataFrame:
        df_applprev = df.filter(pl.col("num_group1")==0)
        df_applprev = df_applprev.drop(["num_group1"])
        features_drop = self.features_df[self.features_df["include"]==0].index.to_list()
        features_drop = [f"{item}_{i}" for item in features_drop for i in range(1, 4)]
        df_applprev = df_applprev.drop(features_drop)
        return df_applprev

    def _process_formulas_all(self, df, features):
        expr_list = []
        new_cols = {}
        for col in features:
            exprs, new_cols_i = self._process_formulas(col, self.features_df.loc[col])
            expr_list.extend(exprs)
            new_cols.update(new_cols_i)
        df = df.with_columns(expr_list)
        df = df.with_columns(
            [pl.col(i).dt.total_days() for i in df.columns if df[i].dtype == pl.Duration]
        )
        return df

    def _process_aggregations_all(self, df, suffix: str = ""):
        features = {i: "_".join(i.split("_")[:-1]) for i in df.columns if i not in self.cols_keep}
        expr_list = []
        for col, col_original in features.items():
            expr_list.extend(self._process_aggregations(col, self.features_df.loc[col_original], suffix=suffix))
        expr_list.append(pl.col(COL_ID).count().alias(f"{COL_ID}_1_count"))
        if self.name in ["creditbureau1", "creditbureau2", "taxregistry1"]:
            expr_list.append(pl.col(COL_ID).count().alias(f"{self.source.replace('*', '')}_1"))
        df = df.group_by(COL_ID).agg(expr_list).sort(COL_ID)
        return df

    def _process_formulas(self, col: str, values_dict: dict) -> list[pl.Expr]:
        """
        Process formulas for a single feature.

        Arguments:
            col: Feature name.
            values_dict: A dictionary containing feature properties.
        Returns:
            list: A list of expressions.
        """
        expr_list = []
        new_cols = {}
        for i in range(1, 4):
            formula_key = f"formula1{i}"
            if not values_dict[formula_key]:
                continue
            expr = Utils.prepare_formula(values_dict, col, i)
            expr_list.append(expr.alias(f"{col}_{i}"))
            new_cols[f"{col}_{i}"] = col
        return expr_list, new_cols

    def _process_aggregations(self, col, value_dict: dict, suffix: str = "") -> list[pl.Expr]:
        """
        Process aggregations for a single feature.

        Arguments:
            value_dict: A dictionary containing feature properties.
        Returns:
            list: A list of aggregations.
        """
        if not value_dict["agg"]:
            return []

        expr_list = []
        funcs = value_dict["agg"].replace(" ", "").split(",")
        expr_list.extend([Utils.agg_process(col, func).alias(f"{col}_{func}{suffix}") for func in funcs])
        # if value_dict["agg"] == "dummy":
        #     expr_list = [i.cast(pl.Int8) for i in expr_list]
        expr_list = [i for i in expr_list if i is not None]
        return expr_list

class BaseDataLoader:
    """
    A class to load the base data for the model.
    """
    def __init__(self, df_props):
        """
        Initialize the BaseDataLoader.
        """
        self.df_props = df_props

    def load_data(self, test: bool, case_ids: list) -> pl.DataFrame:
        """
        Load data based on mode.
        Arguments:
            test: A flag to indicate if the data should be loaded in test mode.
            case_ids: A list of case IDs to filter the data.
        Returns:
            df: polars.DataFrame with base data
        """
        batch_reader = BatchReader(self.df_props, test)
        dfs = []
        for  df, source in batch_reader.batches(case_ids):
            dfs.append(df)
        df = pl.concat(dfs, how="vertical_relaxed")

        if case_ids is not None:
            df = df.filter(pl.col(COL_ID).is_in(case_ids))
        if test:
            df = df.with_columns(pl.lit(0).alias(COL_TARGET))

        return df

class FinalDataProcessor:
    """
    A class to process the final merged dataset.
    """
    def __init__(self, features_df: dict) -> None:
        """
        Initialize the FinalDataProcessor.

        Arguments:
            features_df: A dictionary containing features and their respective processing information.
        """
        self.features_df = features_df

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Process the final merged dataset.

        Arguments:
            df: The DataFrame with the final, merged data.
        Returns:
            pl.DataFrame: The processed DataFrame.
        """
        # Coalesce birthdate columns
        df = df.with_columns(
            pl.coalesce(pl.col("birthdate_574D_1_staticcb0"), pl.col("birth_259D_1_person1")).alias("birthdate_1_final0")
        )
        df = df.with_columns(
            pl.coalesce(pl.col("employedfrom_700D_1_applprev1"), pl.col("empl_employedfrom_271D_1_person1")).alias("employed_length_1_final0")
        )
        df = df.with_columns(
            pl.coalesce(pl.col("maininc_215A_1_static0"), pl.col("mainoccupationinc_384A_1_person1")).alias("mainoccupationinc_1_final0")
        )
        df = df.drop([
            "birthdate_574D_1_staticcb0",
            "birthdate_87D_1_person1",
            "employedfrom_700D_1_applprev1",
            "empl_employedfrom_271D_1_person1",
            "maininc_215A_1_static0",
            "mainoccupationinc_384A_1_person1",
        ])
        # Generate features
        df = df.with_columns(
            (pl.col("mainoccupationinc_1_final0") + pl.col("lastotherinc_902A_1_static0").fill_null(0)).alias("totalinc_1_final0")
        )
        # Fill nans
        case_id_cols = [i for i in df.columns if f"{COL_ID}_" in i]
        df = df.with_columns(
            pl.col(case_id_cols).fill_null(0)
        )
        # Postprocess
        df = df.drop(["MONTH"])
        df = Utils.change_dtypes(df, [pl.String], pl.Categorical)
        df = df.with_columns(pl.when(~(pl.selectors.numeric().is_infinite())).then(pl.selectors.numeric()))
        df = df.sort([COL_WEEK, COL_ID])
        return df
    

class DataProcessor1:
    """
    A utility class for processing data.
    It loads, processes, and prepares datasets for training and testing,
    handling feature engineering, aggregation, and encoding as specified by the dataset's properties.
    """
    def __init__(self, name, version: str = "1") -> None:
        """
        Initializes the DataProcessor with a specific name and loads the dataset properties and features information.

        Arguments:
            name (str): A unique name for the data Utils instance, used for identifying processed datasets.
        """
        self.name = name
        self.path_data = f"{PATH_DATA_PROC}/{self.name}"
        if not KAGGLE:
            utils_old.create_folder(self.path_data)

        # Dict with dataframes to process
        with open(os.path.join(PATH_FEATURES, f"dfs_props_{version}.pkl"), 'rb') as handle:
            self.dfs_props = pickle.load(handle)

        # Dataframe with features information
        self.features_df = utils_old.get_features_df(self.dfs_props, version)

    def get_data(self, test: bool = False, fit: bool = False, load: bool = False, case_ids = None, save_to_disc: bool = True) -> pd.DataFrame:
        """
        Retrieves and processes the dataset based on specified conditions and configurations.

        Arguments:
            test (bool): Specifies whether to load test data.
            fit (bool): Specifies whether to fit the transformer.
            load (bool): Specifies whether to load pre-processed data from disk.
            case_ids (list | None): A list of case IDs to filter the data. If None, no filtering is applied.
        Returns:
            pd.DataFrame: A DataFrame containing processed features and target variables.
        """
        if load:
            df = self.load_data()
            print(f"Number of columns: {len(df.columns)}")
            return df

        # Get base data
        base_data_loader = BaseDataLoader(self.dfs_props["base"])
        df = base_data_loader.load_data(test, case_ids)
        # Get other data
        pbar = tqdm(self.dfs_props.items())
        for name, props in pbar:
            if name == "base":
                continue
            pbar.set_description(name)
            features_df_i = self.features_df[self.features_df["source_group"]==name]
            batch_data_handler = BatchDataHandler(name, props, features_df_i, fit, test)
            df = batch_data_handler.transform(df, case_ids)

        # Process all data
        final_data_processor = FinalDataProcessor(self.features_df)
        df = final_data_processor.transform(df)

        print(f"Number of columns: {len(df.columns)}")

        if save_to_disc:
            self.save_data(df)

        df = df.to_pandas()

        for i in ["a", "b", "c"]:
            col = f"tax_registry_{i}_1_1_taxregistry1"
            if col not in df.columns:
                df[col] = np.nan

        for col in ["credit_bureau_a_2__1_creditbureau2", "credit_bureau_b_2_1_creditbureau2",  'credit_bureau_a_1__1_creditbureau1', 'credit_bureau_b_1_1_creditbureau1']:
            if col not in df.columns:
                df[col] = np.nan

        return df

    def save_data(self, df: pl.DataFrame) -> None:
        """
        Saves the processed DataFrame to the specified path as a CSV file.

        Arguments:
            df (pd.DataFrame): The DataFrame to save.
        """
        path = os.path.join(self.path_data, "data.parquet")
        df.write_parquet(path)

    def load_data(self) -> pd.DataFrame:
        """
        Loads a DataFrame from a specified path and applies categorical encoding to object columns.

        Returns:
            pd.DataFrame: The loaded and processed DataFrame.
        """
        path = os.path.join(self.path_data, "data.parquet")
        df = pl.read_parquet(path)
        df = df.to_pandas()
        return df