"""Data processing: version 4."""

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
from . import utils

VERSION = "4"


class Utils:
    """
    Class for utility functions to process data.
    """
    @staticmethod
    def cast_dtypes(df: pl.DataFrame, schema: dict):
        """
        Cast the data types of the columns in the DataFrame according to the schema.
        Args:
            df: polars.DataFrame, DataFrame with data
            schema: dict, dictionary with column names as keys and polars.DataType as values
        Returns:
            df: polars.DataFrame, DataFrame with casted data types
        """
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
    def change_dtypes(df: pl.DataFrame, dtype_in: list[pl.DataType], dtype_out: pl.DataType) -> pl.DataFrame:
        """
        Change the data types of the columns in the DataFrame.

        Args:
            df: DataFrame with data
            dtype_in: list of polars.DataType, data types to change
            dtype_out new data type
        Returns:
            df: DataFrame with changed data types
        """
        df = df.with_columns(
            [
                pl.col(col).cast(dtype_out)
                for col
                in df.select(pl.selectors.by_dtype(*dtype_in)).columns
            ]
        )
        return df

    @staticmethod
    def downcast(df: pl.DataFrame) -> pl.DataFrame:
        """
        Downcast the data types of the columns in the DataFrame.

        Args:
            df: DataFrame with data
        Returns:
            df: DataFrame with downcasted data types
        """
        df = df.with_columns(pl.col(pl.NUMERIC_DTYPES).shrink_dtype())
        df = df.with_columns(pl.col(COL_ID).cast(pl.Int64))
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
            "ewmmean": lambda: pl.col(col).ewm_mean(com=1).last(),
        }

        if oper in operations:
            return operations[oper]()
        else:
            raise ValueError(f"Operation '{oper}' is not supported.")

    @staticmethod
    def match_cols(dfs: list) -> list:
        """
        Match the columns of the DataFrames.

        Args:
            dfs: list of DataFrames with data
        Returns:
            dfs: list of DataFrames with matched columns
        """
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
    def __init__(self, df_props: dict, test: bool, batch_size: int = 10**7):
        """
        Initializes the BatchReader with dataset properties, mode, and batch size.

        Args:
            df_props: Properties of the dataframe to read, must include 'path' and 'schema'.
            test: Flag indicating whether to read data in test mode. False means train mode.
            batch_size: The number of rows per batch. Should be set according to memory limits.
        """
        self.df_props = df_props
        self.mode = "train" if not test else "test"
        self.batch_size = batch_size

    def _match_cols(self, df: pl.DataFrame, file_name: str) -> pl.DataFrame:
        """
        Matches the columns of the DataFrame with the schema of the dataset.
        (Needed for concatenation of different sources of data).

        Arguments:
            df: dataframe with data
            file_name: name of file
        
        Returns:
            pl.DataFrame: DataFrame with matched columns.
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

    def batches(self, case_ids: list) -> Iterator[pl.DataFrame]:
        """
        Yields data batches from the dataset group, ensuring continuity of case IDs across batches.

        This method reads data in chunks defined by `batch_size`, matches data schema,
        matches columns from different sources, and handles cases where a
        case ID spans multiple batches, ensuring that all data related to a single case ID is processed
        together.

        Arguments:
            case_ids: list of case ids to filter the data.

        Returns:
            An iterator over the data batches.
        """
        str_dict = self.df_props["structure"]
        for file_name, val in str_dict.items():
            paths_i = glob(utils.gen_file_path(file_name, self.mode))
            paths_i = utils.sort_paths(paths_i)
            for path in paths_i:
                df_lazy = pl.scan_parquet(path)
                case_ids_all = df_lazy.select(COL_ID).collect()[COL_ID].to_list()
                n = len(case_ids_all)
                n_batches = int(np.ceil(n/self.batch_size))

                if n == 0:
                    df = df_lazy.collect()
                    df = Utils.cast_dtypes(df, str_dict[file_name]["schema"])
                    df = self._match_cols(df, file_name)
                    if case_ids is not None:
                        df = df.filter(pl.col(COL_ID).is_in(case_ids))
                    yield df, file_name

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
    TEST_BATCH_SIZE = 10**6
    MERGE_COLS =  [COL_TARGET, COL_DATE, COL_WEEK]
    MERGE_COLS_S0 = ["mainoccupationinc_384A_1_person1"]
    MERGE_COLS_S1 = ["credamount_770A_1_static0", "monthsannuity_845L_1_static0", "credtype_322L_1_static0", "maininc_215A_1_static0", "maininc_215A_2_static0",]

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
        merge_cols += self.MERGE_COLS_S0 if self.name != "person1" else []
        merge_cols += self.MERGE_COLS_S1 if self.name not in ["person1", "static0"] else []
        df = df.join(df_init.select(merge_cols), on=COL_ID, how="inner")
        feature_engineer = FeatureEngineer(self.name, source, self.df_props, self.features_df, batch_num, self.fit, merge_cols)
        df_agg = feature_engineer.transform(df)
        df_agg = df_agg.drop([i for i in merge_cols if i != COL_ID])
        # features_list = ["_".join(i.split("_")[:-1]) for i in FEATURES_LIST]
        # df_agg = df_agg.select([i for i in df_agg.columns if i in features_list] + [COL_ID])
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
            cols_keep: Columns to keep in the final DataFrame.
        """
        self.name = name
        self.source = source

        self.df_props = df_props
        self.features_df = features_df

        self.batch_num = batch_num

        self.fit = fit
        self.cols_keep = cols_keep + ["num_group1", "num_group2", "active"]

        self.funcs_map = {
            "applprev1": self._process_applprev1,
            "applprev2": self._process_applprev2,
            "creditbureau1": self._process_creditbureau1,
            "creditbureau2": self._process_creditbureau2,
            "debitcard1": self._process_debitcard1,
            "deposit1": self._process_deposit1,
            "other1": self._process_other1,
            "person1": self._process_person1,
            "person2": self._process_person2,
            "static0": self._process_static0,
            "staticcb0": self._process_staticcb0,
            "taxregistrya1": self._process_taxregistrya1,
            "taxregistryb1": self._process_taxregistryb1,
            "taxregistryc1": self._process_taxregistryc1,

        }

        self.periodicityofpmts_997L_map = {
            "Полугодовые платежи - 180 дней": 180,
            "В день истечения срока кредитного договора": None,
            "Ежеквартальные платежи - 90 дней": 90,
            "Ежемесячные платежи - 30 дней": 30,
            "Взносы с нерегулярной периодичностью": None,
        }

    def transform(self, df: pl.DataFrame, agg: bool = True) -> pl.DataFrame:
        """
        Perform feature engineering on the dataset.

        Arguments:
            df: The DataFrame with the batch data.
            agg: Specifies if the data should be aggregated or returned as is.
        Returns:
            pl.DataFrame: The transformed DataFrame.
        """
        df = Utils.change_dtypes(df, [pl.Categorical, pl.Boolean], pl.String)
        df = Utils.handle_cat(df)
        df = self._handle_dates(df)
        features = [i for i in df.columns if i not in self.cols_keep]
        df = self._process_formulas_all(df)
        df = df.drop(features)

        depth = int(self.name[-1])
        if (depth > 0) & agg:
            dfs = []

            # Handle person1
            if self.name == "person1":
                dfi, df = self._handle_person1(df)
                dfs.append(dfi)

            if self.name == "applprev1":
                dfi = self._handle_applprev1(df)
                dfs.append(dfi)

            if "creditbureau" in self.name:
                dfx = df.filter(pl.col("active"))
                dfx = dfx.select(dfx.select(pl.col(pl.NUMERIC_DTYPES)).columns)
                dfi = self._process_aggregations_all(dfx, suffix="_active")
                dfs.append(dfi)

                dfx = df.filter(pl.col("active"))
                dfx = dfx.select(dfx.select(pl.col(pl.NUMERIC_DTYPES)).columns)
                dfi = self._process_aggregations_all(dfx, suffix="_closed")
                dfs.append(dfi)

            if "taxregistry" in self.name:
                if "taxregistrya1" in self.name:
                    dfx = df.filter(pl.col("recorddate_4527225D_1")>=-30)
                if "taxregistryb1" in self.name:
                    dfx = df.filter(pl.col("deductiondate_4917603D_1")>=-30)
                if "taxregistryc1" in self.name:
                    dfx = df.filter(pl.col("pmtamount_36A_1")>=-30)
                dfi = self._process_aggregations_all(dfx, suffix="_1month")
                dfs.append(dfi)

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
            if (df[col].dtype == pl.Date) & (col != COL_DATE):
                expr_ls.append((pl.col(col) - pl.col(COL_DATE)).dt.total_days().alias(col))
        df = df.with_columns(expr_ls)
        return df
    
    def _handle_contracts_cb(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Merge active and closed contracts into a single flat DataFrame.

        Arguments:
            df: polars.DataFrame with data.
        Returns:
            df_new: polars.DataFrame with active and closed contracts merged 
                    and 'acitve' column indicating the status of a contract.
        """
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

        if self.name == "creditbureau1":
            col_sort = "dateofcredend_all_1"
        elif self.name == "creditbureau2":
            col_sort = "pmts_month_all_1"
        df_new = df_new.sort(col_sort)
        return df_new
    
    def _handle_person1(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Divide the data into two DataFrames: one for num_group1=0 (applicant) and one for the rest.

        Arguments:
            df: polars.DataFrame with data.
        Returns:
            df_persons: polars.DataFrame with data for the applicant of depth 0.
            df: polars.DataFrame with data for other persons in the application of depth 1.
        """
        df_persons = df.filter(pl.col("num_group1")==0)
        df_persons = df_persons.drop(["num_group1"])
        features_drop = self.features_df[self.features_df["include"]==0].index.to_list()
        features_drop = [f"{item}_{i}" for item in features_drop for i in range(1, 4)]
        df_persons = df_persons.drop(features_drop)
        df = df.filter(pl.col("num_group1")!=0)
        return df_persons, df
    
    def _handle_applprev1(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Divide the data into two DataFrames: one for num_group1=0 (last application) and one for the rest.

        Arguments:
            df: polars.DataFrame with data.
        Returns:
            df_applprev: polars.DataFrame with data for the last application.
            df: polars.DataFrame with data for other applications.
        """
        df_applprev = df.filter(pl.col("num_group1")==0)
        features_drop = self.features_df[self.features_df["include"]==0].index.to_list()
        features_drop = [f"{item}_{i}" for item in features_drop for i in range(1, 4)]
        features_drop += ["num_group1"]
        df_applprev = df_applprev.drop(features_drop)
        return df_applprev

    def _process_formulas_all(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Process all formulas for the dataset.

        Arguments:
            df: polars.DataFrame with data.
        Returns:
            df: polars.DataFrame with processed data.
        """
        return self.funcs_map[self.name](df)

    def _process_applprev1(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Process previous applications data.

        Arguments:
            df: polars.DataFrame with data.
        Returns:
            df: polars.DataFrame with processed data.
        """
        cols_drop = [
        ]
        df = df.rename({col: f"{col}_1" for col in df.columns if col not in self.cols_keep+cols_drop})
        df = df.with_columns(
            pl.coalesce(pl.col("dtlastpmt_581D_1"), pl.col("dtlastpmtallstes_3545839D_1")).alias("dtlastpmt_581D_2"),
        )
        df = df.with_columns(
            (pl.col("dtlastpmt_581D_2") - pl.col("approvaldate_319D_1")).alias("dtlastpmt_581D_3"),   
            (pl.col("creationdate_885D_1") - pl.col("approvaldate_319D_1")).alias("creationdate_885D_2"),   
            (pl.col("dtlastpmt_581D_2") - pl.col("firstnonzeroinstldate_307D_1")).alias("dtlastpmt_581D_4"),
            (pl.col("dtlastpmt_581D_2") - pl.col("creationdate_885D_1")).alias("dtlastpmt_581D_5"),
            (pl.col("creationdate_885D_1") - pl.col("employedfrom_700D_1")).alias("creationdate_885D_3"),   
            (pl.col("dtlastpmt_581D_2") - pl.col("approvaldate_319D_1")).alias("dtlastpmt_581D_2"),   
        )
        df = df.with_columns(
            (pl.col("mainoccupationinc_437A_1").fill_null(0) + pl.col("byoccupationinc_3656910L_1").fill_null(0)).alias("mainoccupationinc_437A_2"),
        )
        df = df.with_columns(
            (pl.col("mainoccupationinc_437A_1").fill_null(pl.col("mainoccupationinc_384A_1_person1"))).alias("mainoccupationinc_437A_3"),
            (pl.col("mainoccupationinc_437A_2").fill_null(pl.col("mainoccupationinc_384A_1_person1"))).alias("mainoccupationinc_437A_4"),
        )
        df = df.with_columns(
            pl.when(pl.col("actualdpd_943P_1")>1).then(1).otherwise(0).alias("actualdpd_943P_5"),
            pl.when(pl.col("actualdpd_943P_1")>3).then(1).otherwise(0).alias("actualdpd_943P_2"),
            pl.when(pl.col("actualdpd_943P_1")>7).then(1).otherwise(0).alias("actualdpd_943P_3"),
            pl.when(pl.col("actualdpd_943P_1")>15).then(1).otherwise(0).alias("actualdpd_943P_4"),
            pl.when(pl.col("maxdpdtolerance_577P_1")>1).then(1).otherwise(0).alias("maxdpdtolerance_577P_5"),
            pl.when(pl.col("maxdpdtolerance_577P_1")>3).then(1).otherwise(0).alias("maxdpdtolerance_577P_2"),
            pl.when(pl.col("maxdpdtolerance_577P_1")>7).then(1).otherwise(0).alias("maxdpdtolerance_577P_3"),
            pl.when(pl.col("maxdpdtolerance_577P_1")>15).then(1).otherwise(0).alias("maxdpdtolerance_577P_4"),
        )
        
        df = df.with_columns(
            (pl.col("annuity_853A_1") / pl.col("monthsannuity_845L_1_static0")).alias("annuity_853A_4"),
            (pl.col("mainoccupationinc_437A_3") / pl.col("childnum_21L_1")).alias("mainoccupationinc_437A_5"),
            (pl.col("mainoccupationinc_437A_4") / pl.col("childnum_21L_1")).alias("mainoccupationinc_437A_6"),
            (pl.col("mainoccupationinc_437A_4") / pl.col("maininc_215A_1_static0")).alias("mainoccupationinc_437A_7"),
            (pl.col("credacc_actualbalance_314A_1") + pl.col("mainoccupationinc_437A_4")).alias("credacc_actualbalance_314A_2"),
            (pl.col("credacc_maxhisbal_375A_1") / pl.col("credacc_credlmt_575A_1")).alias("credacc_maxhisbal_375A_2"),
            (pl.col("credacc_minhisbal_90A_1") / pl.col("credacc_credlmt_575A_1")).alias("credacc_minhisbal_90A_2"),
            (pl.col("credamount_590A_1") / pl.col("credamount_770A_1_static0")).alias("credamount_590A_4"),
            (pl.col("downpmt_134A_1") / pl.col("mainoccupationinc_437A_1")).alias("downpmt_134A_2"),
            (pl.col("pmtnum_8L_1") / pl.col("tenor_203L_1")).alias("pmtnum_8L_2"),

            (pl.col("credacc_credlmt_575A_1") / pl.col("credacc_actualbalance_314A_1")).alias("credacc_credlmt_575A_3"),
            (pl.col("currdebt_94A_1") / pl.col("mainoccupationinc_437A_1")).alias("currdebt_94A_2"),
            (pl.col("outstandingdebt_522A_1") / pl.col("mainoccupationinc_437A_1")).alias("outstandingdebt_522A_2"),
            
            ((pl.col("annuity_853A_1") * pl.col("tenor_203L_1") / pl.col("credamount_590A_1") - 1) / pl.col("tenor_203L_1")).alias("annuity_853A_5"),
        )

        cols_amount = [
            "annuity_853A_1",
            "credacc_actualbalance_314A_1",
            "credacc_credlmt_575A_1",
            "credacc_maxhisbal_375A_1",
            "credacc_minhisbal_90A_1",
            "credamount_590A_1",
            "currdebt_94A_1",
            "downpmt_134A_1",
            "mainoccupationinc_437A_1",
            "outstandingdebt_522A_1",
        ]
        expr_list = []
        for col in cols_amount:
            expr_list.append((pl.col(col)/pl.col("credamount_770A_1_static0")).alias(f"{col}1"))
            expr_list.append((pl.col(col)/pl.col("maininc_215A_2_static0")).alias(f"{col}2"))
            expr_list.append((pl.col(col)/pl.col("credamount_590A_1")).alias(f"{col}3"))
            expr_list.append((pl.col(col)/pl.col("mainoccupationinc_437A_1")).alias(f"{col}4"))
        df = df.with_columns(expr_list)

        df = df.sort("creationdate_885D_1")

        return df

    def _process_applprev2(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Process previous applications data (depth 2).

        Arguments:
            df: polars.DataFrame with data.
        Returns:
            df: polars.DataFrame with processed data.
        """
        cols_drop = []
        df = df.rename({col: f"{col}_1" for col in df.columns if col not in self.cols_keep+cols_drop})
        return df
    
    def _process_creditbureau1(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Process credit bureau data (credits).

        Arguments:
            df: polars.DataFrame with data.
        Returns:
            df: polars.DataFrame with processed data.
        """
        cols_drop = [
            "dpdmaxdateyear_896T",
            "dpdmaxdateyear_596T",
            "overdueamountmaxdateyear_994T",
            "overdueamountmaxdateyear_2T",
            "dpdmaxdatemonth_442T",
            "dpdmaxdatemonth_89T",
            "overdueamountmaxdatemonth_365T",
            "overdueamountmaxdatemonth_284T",
            "credlmt_1052A",
            "instlamount_892A",
            "periodicityofpmts_997M",
            "periodicityofpmts_997L",
            "residualamount_3940956A",
        ]
        df = df.rename({col: f"{col}_1" for col in df.columns if col not in self.cols_keep+cols_drop})

        df = df.with_columns(
            (pl.col("dateofcredend_289D_1") / pl.col("numberofcontrsvalue_258L_1")).alias("dateofcredend_289D_3"),
            (pl.col("dateofcredend_353D_1") / pl.col("numberofcontrsvalue_358L_1")).alias("dateofcredend_353D_3"),
            (pl.col("dateofcredend_289D_1") - pl.col("dateofcredstart_181D_1")).alias("dateofcredend_289D_2"),
            (pl.col("dateofcredend_353D_1") - pl.col("dateofcredstart_739D_1")).alias("dateofcredend_353D_2"),
            (pl.col("numberofoverdueinstlmaxdat_148D_1") - pl.col("dateofcredstart_739D_1")).alias("numberofoverdueinstlmaxdat_148D_2"),
            (pl.col("numberofoverdueinstlmaxdat_641D_1") - pl.col("dateofcredstart_181D_1")).alias("numberofoverdueinstlmaxdat_641D_2"),
            (pl.col("overdueamountmax2date_1002D_1") - pl.col("dateofcredstart_739D_1")).alias("overdueamountmax2date_1002D_2"),
            (pl.col("overdueamountmax2date_1142D_1") - pl.col("dateofcredstart_181D_1")).alias("overdueamountmax2date_1142D_2"),
        )

        df = df.with_columns(
            pl.when(pl.col("dpdmax_139P_1")>1).then(1).otherwise(0).alias("dpdmax_139P_5"),
            pl.when(pl.col("dpdmax_139P_1")>3).then(1).otherwise(0).alias("dpdmax_139P_2"),
            pl.when(pl.col("dpdmax_139P_1")>7).then(1).otherwise(0).alias("dpdmax_139P_3"),
            pl.when(pl.col("dpdmax_139P_1")>15).then(1).otherwise(0).alias("dpdmax_139P_4"),
            pl.when(pl.col("dpdmax_757P_1")>1).then(1).otherwise(0).alias("dpdmax_757P_5"),
            pl.when(pl.col("dpdmax_757P_1")>3).then(1).otherwise(0).alias("dpdmax_757P_2"),
            pl.when(pl.col("dpdmax_757P_1")>7).then(1).otherwise(0).alias("dpdmax_757P_3"),
            pl.when(pl.col("dpdmax_757P_1")>15).then(1).otherwise(0).alias("dpdmax_757P_4"),
        )
        

        df = df.with_columns(
            pl.coalesce(pl.col("totalamount_996A_1"), pl.col("credlmt_935A_1")).alias("totalamount_996A_2"),
            pl.coalesce(pl.col("totalamount_6A_1"), pl.col("credlmt_230A_1")).alias("totalamount_6A_2"),
        )

        df = df.with_columns(
            (pl.date(pl.col("dpdmaxdateyear_896T"), pl.col("dpdmaxdatemonth_442T"), 1) - pl.col("date_decision")).dt.total_days().alias("dpdmaxdatemonth_442T_1"),
            (pl.date(pl.col("dpdmaxdateyear_596T"), pl.col("dpdmaxdatemonth_89T"), 1) - pl.col("date_decision")).dt.total_days().alias("dpdmaxdatemonth_89T_1"),
            (pl.date(pl.col("overdueamountmaxdateyear_2T"), pl.col("overdueamountmaxdatemonth_365T"), 1) - pl.col("date_decision")).dt.total_days().alias("overdueamountmaxdatemonth_365T_1"),
            (pl.date(pl.col("overdueamountmaxdateyear_994T"), pl.col("overdueamountmaxdatemonth_284T"), 1) - pl.col("date_decision")).dt.total_days().alias("overdueamountmaxdatemonth_284T_1"),
        )
        df = df.with_columns(
            (pl.col("numberofoverdueinstlmax_1039L_1") / pl.col("numberofinstls_320L_1")).alias("numberofoverdueinstlmax_1039L_2"),
            (pl.col("numberofoverdueinstlmax_1151L_1") / pl.col("numberofinstls_229L_1")).alias("numberofoverdueinstlmax_1151L_2"),
            (pl.col("numberofoutstandinstls_520L_1") / pl.col("numberofinstls_229L_1")).alias("numberofoutstandinstls_520L_2"),
            (pl.col("numberofoutstandinstls_59L_1") / pl.col("numberofinstls_320L_1")).alias("numberofoutstandinstls_59L_2"),
            (pl.col("dpdmax_139P_1")/pl.col("numberofoverdueinstls_725L_1")).alias("dpdmax_139P_5"),
            (pl.col("dpdmax_757P_1") / pl.col("numberofoverdueinstls_834L_1")).alias("dpdmax_757P_5"),
            (pl.col("debtoverdue_47A_1") / pl.col("debtoutstand_525A_1")).alias("debtoverdue_47A_2"),
        )
        df = df.with_columns(
            (pl.col("monthlyinstlamount_332A_1") / pl.col("monthsannuity_845L_1_static0")).alias("monthlyinstlamount_332A_2"),
            (pl.col("monthlyinstlamount_674A_1") / pl.col("monthsannuity_845L_1_static0")).alias("monthlyinstlamount_674A_2"),
        )

        cols_amount = [
            "credlmt_230A_1",
            "credlmt_935A_1",
            "debtoutstand_525A_1",
            "debtoverdue_47A_1",
            "instlamount_768A_1",
            "instlamount_852A_1",
            "monthlyinstlamount_332A_1",
            "monthlyinstlamount_674A_1",
            "outstandingamount_354A_1",
            "outstandingamount_362A_1",
            "overdueamount_31A_1",
            "overdueamount_659A_1",
            "overdueamountmax_155A_1",
            "overdueamountmax_35A_1",
            "overdueamountmax2_14A_1",
            "overdueamountmax2_398A_1",
            "residualamount_488A_1",
            "residualamount_856A_1",
            "totalamount_6A_1",
            "totalamount_996A_1",
            "totaldebtoverduevalue_178A_1",
            "totaldebtoverduevalue_718A_1",
            "totaloutstanddebtvalue_39A_1",
            "totaloutstanddebtvalue_668A_1",
            "totalamount_996A_2",
            "totalamount_6A_2",
        ]
        cols_amount_active = [
            "credlmt_935A_1",
            "debtoutstand_525A_1",
            "debtoverdue_47A_1",
            "instlamount_768A_1",
            "monthlyinstlamount_332A_1",
            "outstandingamount_362A_1",
            "overdueamount_659A_1",
            "overdueamountmax_155A_1",
            "overdueamountmax2_14A_1",
            "residualamount_856A_1",
            "totalamount_996A_1",
            "totaldebtoverduevalue_178A_1",
            "totaloutstanddebtvalue_39A_1",
        ]
        cols_amount_closed = [
            "credlmt_230A_1",
            "instlamount_852A_1",
            "monthlyinstlamount_674A_1",
            "outstandingamount_354A_1",
            "overdueamount_31A_1",
            "overdueamountmax_35A_1",
            "overdueamountmax2_398A_1",
            "residualamount_488A_1",
            "totalamount_6A_1",
            "totaldebtoverduevalue_718A_1",
            "totaloutstanddebtvalue_668A_1",
        ]
        expr_list = []
        for col in cols_amount:
            expr_list.append((pl.col(col)/pl.col("credamount_770A_1_static0")).alias(f"{col}1"))
            expr_list.append((pl.col(col)/pl.col("maininc_215A_2_static0")).alias(f"{col}2"))
        for col in cols_amount_active:
            expr_list.append((pl.col(col)/pl.col("totalamount_996A_2")).alias(f"{col}3"))
        for col in cols_amount_closed:
            expr_list.append((pl.col(col)/pl.col("totalamount_6A_2")).alias(f"{col}3"))
        df = df.with_columns(expr_list)

        df = self._handle_contracts_cb(df)
        return df
    
    def _process_creditbureau2(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Process credit bureau data (installments).

        Arguments:
            df: polars.DataFrame with data.
        Returns:
            df: polars.DataFrame with processed data.
        """
        cols_drop = [
            "pmts_year_507T",
            "pmts_year_1139T",
            "pmts_month_158T",
            "pmts_month_706T",
        ]
        df = df.rename({col: f"{col}_1" for col in df.columns if col not in self.cols_keep+cols_drop})
        df = df.with_columns(
            (pl.date(pl.col("pmts_year_507T"), pl.col("pmts_month_158T"), 1) - pl.col("date_decision")).dt.total_days().alias("pmts_month_158T_1"),
            (pl.date(pl.col("pmts_year_1139T"), pl.col("pmts_month_706T"), 1) - pl.col("date_decision")).dt.total_days().alias("pmts_month_706T_1"),
        )
        df = df.with_columns(
            pl.when(pl.col("pmts_dpd_1073P_1")>1).then(1).otherwise(0).alias("pmts_dpd_1073P_5"),
            pl.when(pl.col("pmts_dpd_303P_1")>1).then(1).otherwise(0).alias("pmts_dpd_303P_5"),
            pl.when(pl.col("pmts_dpd_1073P_1")>3).then(1).otherwise(0).alias("pmts_dpd_1073P_2"),
            pl.when(pl.col("pmts_dpd_303P_1")>3).then(1).otherwise(0).alias("pmts_dpd_303P_2"),
            pl.when(pl.col("pmts_dpd_1073P_1")>7).then(1).otherwise(0).alias("pmts_dpd_1073P_3"),
            pl.when(pl.col("pmts_dpd_303P_1")>7).then(1).otherwise(0).alias("pmts_dpd_303P_3"),
            pl.when(pl.col("pmts_dpd_1073P_1")>15).then(1).otherwise(0).alias("pmts_dpd_1073P_4"),
            pl.when(pl.col("pmts_dpd_303P_1")>15).then(1).otherwise(0).alias("pmts_dpd_303P_4"),
        )
        df = df.with_columns(
            (pl.col("pmts_dpd_1073P_1") * pl.col("pmts_month_158T_1")).alias("pmts_dpd_1073P_5"),
            (pl.col("pmts_dpd_303P_1") * pl.col("pmts_month_158T_1")).alias("pmts_dpd_303P_5"),
            (pl.col("pmts_dpd_1073P_2") * pl.col("pmts_month_158T_1")).alias("pmts_dpd_1073P_6"),
            (pl.col("pmts_dpd_303P_2") * pl.col("pmts_month_158T_1")).alias("pmts_dpd_303P_6"),
            (pl.col("pmts_dpd_1073P_3") * pl.col("pmts_month_158T_1")).alias("pmts_dpd_1073P_7"),
            (pl.col("pmts_dpd_303P_3") * pl.col("pmts_month_158T_1")).alias("pmts_dpd_303P_7"),
            (pl.col("pmts_dpd_1073P_4") * pl.col("pmts_month_158T_1")).alias("pmts_dpd_1073P_8"),
            (pl.col("pmts_dpd_303P_4") * pl.col("pmts_month_158T_1")).alias("pmts_dpd_303P_8"),
        )

        cols_amount = [
            "pmts_overdue_1140A_1",
            "pmts_overdue_1152A_1",
        ]
        expr_list = []
        for col in cols_amount:
            expr_list.append((pl.col(col)/pl.col("credamount_770A_1_static0")).alias(f"{col}1"))
            expr_list.append((pl.col(col)/pl.col("maininc_215A_2_static0")).alias(f"{col}2"))
        df = df.with_columns(expr_list)

        df = self._handle_contracts_cb(df)

        return df
    
    def _process_debitcard1(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Process debit cards data.

        Arguments:
            df: polars.DataFrame with data.
        Returns:
            df: polars.DataFrame with processed data.
        """
        cols_drop = []
        df = df.rename({col: f"{col}_1" for col in df.columns if col not in self.cols_keep+cols_drop})
        cols_amount = [
            "last180dayaveragebalance_704A_1",
            "last180dayturnover_1134A_1",
            "last30dayturnover_651A_1",
        ]
        expr_list = []
        for col in cols_amount:
            expr_list.append((pl.col(col)/pl.col("credamount_770A_1_static0")).alias(f"{col}1"))
            expr_list.append((pl.col(col)/pl.col("maininc_215A_2_static0")).alias(f"{col}2"))
        df = df.with_columns(expr_list)
        return df
    
    def _process_deposit1(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Process deposits data.

        Arguments:
            df: polars.DataFrame with data.
        Returns:
            df: polars.DataFrame with processed data.
        """
        cols_drop = []
        df = df.rename({col: f"{col}_1" for col in df.columns if col not in self.cols_keep+cols_drop})
        cols_amount = [
            "amount_416A_1",
        ]
        expr_list = []
        for col in cols_amount:
            expr_list.append((pl.col(col)/pl.col("credamount_770A_1_static0")).alias(f"{col}1"))
            expr_list.append((pl.col(col)/pl.col("maininc_215A_2_static0")).alias(f"{col}2"))
        df = df.with_columns(expr_list)
        return df
    
    def _process_other1(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Process other data.

        Arguments:
            df: polars.DataFrame with data.
        Returns:
            df: polars.DataFrame with processed data.
        """
        cols_drop = []
        df = df.rename({col: f"{col}_1" for col in df.columns if col not in self.cols_keep+cols_drop})
        df = df.with_columns(
            (pl.col("amtdebitincoming_4809443A_1") - pl.col("amtdebitoutgoing_4809440A_1")).alias("amtdebitincoming_4809443A_2"),
            (pl.col("amtdepositincoming_4809444A_1") - pl.col("amtdepositoutgoing_4809442A_1")).alias("amtdepositincoming_4809444A_2")
        )
        cols_amount = [
            "amtdebitincoming_4809443A_1",
            "amtdebitoutgoing_4809440A_1",
            "amtdepositbalance_4809441A_1",
            "amtdepositincoming_4809444A_1",
            "amtdepositoutgoing_4809442A_1",
            "amtdebitincoming_4809443A_2",
            "amtdepositincoming_4809444A_2",
        ]
        expr_list = []
        for col in cols_amount:
            expr_list.append((pl.col(col)/pl.col("credamount_770A_1_static0")).alias(f"{col}1"))
            expr_list.append((pl.col(col)/pl.col("maininc_215A_2_static0")).alias(f"{col}2"))
        df = df.with_columns(expr_list)
        return df
    
    def _process_person1(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Process applicant and related persons data.

        Arguments:
            df: polars.DataFrame with data.
        Returns:
            df: polars.DataFrame with processed data.
        """
        cols_drop = [
            "contaddr_district_15M",
            "registaddr_district_1083M",
            "empladdr_district_926M",
            "contaddr_zipcode_807M",
            "registaddr_zipcode_184M",
            "empladdr_zipcode_114M",
            "personindex_1023L",
            "isreference_387L",
            "birthdate_87D", 
            "sex_738L",
            "relationshiptoclient_642T",
        ]
        df = df.rename({col: f"{col}_1" for col in df.columns if col not in self.cols_keep+cols_drop})
        df = df.with_columns(
            pl.coalesce(pl.col("birth_259D_1"), pl.col("birthdate_87D")).alias("birth_259D_1"),
            pl.coalesce(pl.col("gender_992L_1"), pl.col("sex_738L")).alias("gender_992L_1"),
            pl.coalesce(pl.col("relationshiptoclient_415T_1"), pl.col("relationshiptoclient_642T")).alias("relationshiptoclient_415T_1"),
        )
        return df
    
    def _process_person2(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Process applicant and related persons data (depth 2).

        Arguments:
            df: polars.DataFrame with data.
        Returns:
            df: polars.DataFrame with processed data.
        """
        cols_drop = [
            "addres_district_368M",
            "empls_employer_name_740M",
            "relatedpersons_role_762T",
            "addres_role_871L",
            "empls_employedfrom_796D",
        ]
        df = df.rename({col: f"{col}_1" for col in df.columns if col not in self.cols_keep+cols_drop})
        return df
    
    def _process_static0(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Process static data.

        Arguments:
            df: polars.DataFrame with data.
        Returns:
            df: polars.DataFrame with processed data.
        """
        cols_drop = [
            "previouscontdistrict_112M",
            "paytype1st_925L",
            "paytype_783L",
        ]
        df = df.rename({col: f"{col}_1" for col in df.columns if col not in self.cols_keep+cols_drop})

        df = df.with_columns(
            (pl.col("lastotherinc_902A_1") + pl.col("maininc_215A_1").fill_null(pl.col("mainoccupationinc_384A_1_person1"))).alias("maininc_215A_2"),
            (pl.col("lastotherlnsexpense_631A_1") + pl.col("monthsannuity_845L_1")).alias("lastotherlnsexpense_631A_2"),
        )
        df = df.with_columns(
            (pl.col("maininc_215A_2") / pl.col("lastotherlnsexpense_631A_2")).alias("maininc_215A_3"),
        )

        expr_list = []
        cols_amount = [
            "annuity_780A_1",
            "annuitynextmonth_57A_1",
            "avgoutstandbalancel6m_4187114A_1",
            "avgpmtlast12m_4525200A_1",
            "credamount_770A_1",
            "currdebt_22A_1",
            "currdebtcredtyperange_828A_1",
            "disbursedcredamount_1113A_1",
            "inittransactionamount_650A_1",
            "lastapprcredamount_781A_1",
            "lastotherinc_902A_1",
            "lastotherlnsexpense_631A_1",
            "maininc_215A_1",
            "maxannuity_159A_1",
            "maxannuity_4075009A_1",
            "maxdebt4_972A_1",
            "maxinstallast24m_3658928A_1",
            "maxlnamtstart6m_4525199A_1",
            "maxoutstandbalancel12m_4187113A_1",
            "maxpmtlast3m_4525190A_1",
            "sumoutstandtotal_3546847A_1",
            "sumoutstandtotalest_4493215A_1",
            "totaldebt_9A_1",
            "totalsettled_863A_1",
            "totinstallast1m_4525188A_1",
        ]
        for col in cols_amount:
            expr_list.append((pl.col(col)/pl.col("credamount_770A_1")).alias(f"{col}1"))
            expr_list.append((pl.col(col)/pl.col("maininc_215A_2")).alias(f"{col}2"))

        df = df.with_columns(expr_list)


        df = df.with_columns(
            (pl.col("actualdpdtolerance_344P_1") / pl.col("avgdbddpdlast24m_3658932P_1")).alias("actualdpdtolerance_344P_2"), #Debt Payment Ratio
            (pl.col("annuity_780A_1") / pl.col("avgpmtlast12m_4525200A_1")).alias("annuity_780A_2"), # Financial Stability Ratio
            (pl.col("avgoutstandbalancel6m_4187114A_1") / pl.col("avglnamtstart24m_4525187A_1")).alias("avgoutstandbalancel6m_4187114A_2"), # Loan Utilization Ratio
            (pl.col("annuitynextmonth_57A_1") - pl.col("annuity_780A_1")).alias("annuitynextmonth_57A_2"), # Annuity change
            (pl.col("avgdbddpdlast3m_4187120P_1") - pl.col("avgdbdtollast24m_4525197P_1")).alias("avgdbddpdlast3m_4187120P_2"), # Change in dpd
            (pl.col("avgdbddpdlast3m_4187120P_1") - pl.col("avgmaxdpdlast9m_3716943P_1")).alias("avgdbddpdlast3m_4187120P_3"), # Change in dpd
            (pl.col("clientscnt3m_3712950L_1") - pl.col("clientscnt12m_3712952L_1")).alias("clientscnt3m_3712950L_2"), # Change in clients same phone number
            (pl.col("cntincpaycont9m_3716944L_1") - pl.col("cntpmts24_3658933L_1")).alias("cntincpaycont9m_3716944L_2"), # Change in income payments
            (pl.col("dtlastpmtallstes_4499206D_1") - pl.col("datelastunpaid_3546854D_1")).alias("dtlastpmtallstes_4499206D_2"), # Time since last unpaid
            (pl.col("lastapplicationdate_877D_1") - pl.col("datelastunpaid_3546854D_1")).alias("lastapplicationdate_877D_2"), # Time since last unpaid
            (pl.col("lastapprdate_640D_1") - pl.col("datelastunpaid_3546854D_1")).alias("lastapprdate_640D_2"), # Time since last unpaid
            (pl.col("maxannuity_159A_1")/pl.col("maxannuity_4075009A_1")).alias("maxannuity_159A_2"), # Max annuity ratio
            (pl.col("maxdbddpdlast1m_3658939P_1")-pl.col("maxdbddpdtollast12m_3658940P_1")).alias("maxdbddpdlast1m_3658939P_2"), # Change in dpd
            (pl.col("maxdbddpdlast1m_3658939P_1")-pl.col("maxdpdfrom6mto36m_3546853P_1")).alias("maxdbddpdlast1m_3658939P_3"), # Change in dpd
            (pl.col("maxdpdlast3m_392P_1")-pl.col("maxdpdlast6m_474P_1")).alias("maxdpdlast3m_392P_2"), # Change in dpd
            (pl.col("maxdpdlast6m_474P_1")-pl.col("maxdpdlast12m_727P_1")).alias("maxdpdlast6m_474P_2"), # Change in dpd
            (pl.col("maxdpdlast12m_727P_1")-pl.col("maxdpdlast24m_143P_1")).alias("maxdpdlast12m_727P_2"), # Change in dpd

        )


        df = df.with_columns(
            pl.coalesce(pl.col("sumoutstandtotal_3546847A_1"), pl.col("sumoutstandtotalest_4493215A_1")).alias("sumoutstandtotal_3546847A_2"),
            pl.coalesce(pl.col("maininc_215A_1"), pl.col("mainoccupationinc_384A_1_person1")).alias("maininc_215A_5"),
            pl.coalesce(pl.col("numinsttopaygr_769L_1"), pl.col("numinstregularpaidest_4493210L_1")).alias("numinsttopaygr_769L_2"),
            pl.coalesce(pl.col("numinstpaidearly3d_3546850L_1"), pl.col("numinstpaidearly3dest_4493216L_1")).alias("numinstpaidearly3d_3546850L_2"),
            pl.coalesce(pl.col("numinstpaidearly5d_1087L_1"), pl.col("numinstpaidearly5dest_4493211L_1"), pl.col("numinstpaidearly5dobd_4499205L_1")).alias("numinstpaidearly5d_1087L_2"),
            pl.coalesce(pl.col("numinstunpaidmax_3546851L_1"), pl.col("numinstunpaidmaxest_4493212L_1")).alias("numinstunpaidmax_3546851L_2"),
            pl.coalesce(pl.col("numinstlsallpaid_934L_1"), pl.col("numinstpaid_4499208L_1")).alias("numinstlsallpaid_934L_2"),
            pl.coalesce(pl.col("numinstregularpaid_973L_1"), pl.col("numinstregularpaidest_4493210L_1")).alias("numinstregularpaid_973L_2"),
        )

        df = df.with_columns(
            (pl.col("totalsettled_863A_1") / pl.col("totaldebt_9A_1")).alias("totalsettled_863A_2"),
            (pl.col("annuity_780A_1") / pl.col("avginstallast24m_3658937A_1")).alias("annuity_780A_4"),
            (pl.col("annuity_780A_1") / pl.col("avgpmtlast12m_4525200A_1")).alias("annuity_780A_5"),
            (pl.col("datelastunpaid_3546854D_1") - pl.col("datefirstoffer_1144D_1")).alias("datelastunpaid_3546854D_2"),
            (pl.col("dtlastpmtallstes_4499206D_1") - pl.col("datelastunpaid_3546854D_1")).alias("dtlastpmtallstes_4499206D_2"),
            (pl.col("inittransactionamount_650A_1") / pl.col("annuity_780A_1")).alias("inittransactionamount_650A_2"),
            (pl.col("eir_270L_1") - pl.col("interestrate_311L_1")).alias("eir_270L_2"),
            (pl.col("interestrategrace_34L_1") / pl.col("interestrate_311L_1")).alias("interestrategrace_34L_2"),
            (pl.col("credamount_770A_1") / pl.col("lastapprcredamount_781A_1")).alias("credamount_770A_3"),
            (pl.col("lastapprdate_640D_1") - pl.col("lastapplicationdate_877D_1")).alias("lastapprdate_640D_2"),
            (pl.col("lastrejectdate_50D_1") - pl.col("lastapplicationdate_877D_1")).alias("lastrejectdate_50D_2"),
            (pl.col("lastrepayingdate_696D_1") / pl.col("datelastunpaid_3546854D_1")).alias("lastrepayingdate_696D_2"),
            (pl.col("maininc_215A_1") / pl.col("mainoccupationinc_384A_1_person1")).alias("maininc_215A_6"),
            (pl.col("annuity_780A_1") / pl.col("maxannuity_159A_1")).alias("annuity_780A_6"),
            (pl.col("currdebt_22A_1") / pl.col("maxdebt4_972A_1")).alias("currdebt_22A_3"),
            (pl.col("annuity_780A_1") / pl.col("maxannuity_4075009A_1")).alias("annuity_780A_7"),
            (pl.col("annuity_780A_1") / pl.col("maxinstallast24m_3658928A_1")).alias("annuity_780A_8"),
            (pl.col("credamount_770A_1") / pl.col("maxlnamtstart6m_4525199A_1")).alias("credamount_770A_4"),
            (pl.col("annuity_780A_1") / pl.col("maxpmtlast3m_4525190A_1")).alias("annuity_780A_9"),
            (pl.col("monthsannuity_845L_1") / pl.col("annuity_780A_1")).alias("monthsannuity_845L_2"),
            (pl.col("numinstlallpaidearly3d_817L_1") / pl.col("numinstlsallpaid_934L_2")).alias("numinstlallpaidearly3d_817L_2"),
            (pl.col("numinstlswithdpd10_728L_1") / pl.col("numinstlsallpaid_934L_2")).alias("numinstlswithdpd10_728L_2"),
            (pl.col("numinstlswithdpd5_4187116L_1") / pl.col("numinstlsallpaid_934L_2")).alias("numinstlswithdpd5_4187116L_2"),
            (pl.col("numinstlswithoutdpd_562L_1") / pl.col("numinstlsallpaid_934L_2")).alias("numinstlswithoutdpd_562L_2"),
            (pl.col("numinstmatpaidtearly2d_4499204L_1") / pl.col("numinstlsallpaid_934L_2")).alias("numinstmatpaidtearly2d_4499204L_2"),
            (pl.col("numinstpaidearly_338L_1") / pl.col("numinstlsallpaid_934L_2")).alias("numinstpaidearly_338L_2"),
            (pl.col("numinstpaidearly3d_3546850L_2") / pl.col("numinstlsallpaid_934L_2")).alias("numinstpaidearly3d_3546850L_3"),
            (pl.col("numinstpaidearly5d_1087L_2") / pl.col("numinstlsallpaid_934L_2")).alias("numinstpaidearly5d_1087L_3"),
            (pl.col("numinstpaidearlyest_4493214L_1") / pl.col("numinstlsallpaid_934L_2")).alias("numinstpaidearlyest_4493214L_2"),
            (pl.col("numinstpaidlastcontr_4325080L_1") / pl.col("numinstlsallpaid_934L_2")).alias("numinstpaidlastcontr_4325080L_2"),
            (pl.col("numinstpaidlate1d_3546852L_1") / pl.col("numinstlsallpaid_934L_2")).alias("numinstpaidlate1d_3546852L_2"),
            (pl.col("numinstregularpaid_973L_2") / pl.col("numinstlsallpaid_934L_2")).alias("numinstregularpaid_973L_3"),
            (pl.col("numinsttopaygr_769L_2") / pl.col("numinstlsallpaid_934L_2")).alias("numinsttopaygr_769L_3"),
            (pl.col("numinstunpaidmax_3546851L_2") / pl.col("numinstlsallpaid_934L_2")).alias("numinstunpaidmax_3546851L_3"),
        )

        df = df.drop(cols_drop)

        return df
    
    def _process_staticcb0(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Process additional static data.

        Arguments:
            df: polars.DataFrame with data.
        Returns:
            df: polars.DataFrame with processed data.
        """
        cols_drop = []
        df = df.rename({col: f"{col}_1" for col in df.columns if col not in self.cols_keep+cols_drop})

        df = df.with_columns(
            pl.coalesce(pl.col("pmtaverage_4527227A_1"), pl.col("pmtaverage_4955615A_1"), pl.col("pmtaverage_3A_1")).alias("pmtaverage_3A_2"),
            pl.coalesce(pl.col("birthdate_574D_1"), pl.col("dateofbirth_337D_1"), pl.col("dateofbirth_342D_1")).alias("birthdate_574D_2"),
            pl.coalesce(pl.col("education_1103M_1"), pl.col("education_88M_1")).alias("education_1103M_2"),
            pl.coalesce(pl.col("maritalst_385M_1"), pl.col("maritalst_893M_1")).alias("maritalst_385M_2"),
            pl.coalesce(pl.col("pmtcount_4527229L_1"), pl.col("pmtcount_4955617L_1"), pl.col("pmtcount_693L_1")).alias("pmtcount_4527229L_2"),
            pl.coalesce(pl.col("assignmentdate_238D_1"), pl.col("assignmentdate_4527235D_1"), pl.col("assignmentdate_4955616D_1")).alias("assignmentdate_238D_2"),
        )
        cols_amount = [
            "pmtaverage_3A_1",
            "pmtaverage_4527227A_1",
            "pmtaverage_4955615A_1",
            "pmtssum_45A_1",
            "contractssum_5085716L_1",
        ]
        expr_list = []
        for col in cols_amount:
            expr_list.append((pl.col(col)/pl.col("credamount_770A_1_static0")).alias(f"{col}1"))
            expr_list.append((pl.col(col)/pl.col("maininc_215A_2_static0")).alias(f"{col}2"))
        df = df.with_columns(expr_list)
        return df
    
    def _process_taxregistrya1(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Process taxregistry data from source a.

        Arguments:
            df: polars.DataFrame with data.
        Returns:
            df: polars.DataFrame with processed data.
        """
        cols_drop = [
            "name_4527232M"
        ]
        df = df.rename({col: f"{col}_1" for col in df.columns if col not in self.cols_keep+cols_drop})
        cols_amount = [
            "amount_4527230A_1",
        ]
        expr_list = []
        for col in cols_amount:
            expr_list.append((pl.col(col)/pl.col("credamount_770A_1_static0")).alias(f"{col}1"))
            expr_list.append((pl.col(col)/pl.col("maininc_215A_2_static0")).alias(f"{col}2"))
        df = df.with_columns(expr_list)
        df = df.sort("recorddate_4527225D_1")
        return df
    
    def _process_taxregistryb1(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Process taxregistry data from source b.

        Arguments:
            df: polars.DataFrame with data.
        Returns:
            df: polars.DataFrame with processed data.
        """
        cols_drop = [
            "name_4917606M"
        ]
        df = df.rename({col: f"{col}_1" for col in df.columns if col not in self.cols_keep+cols_drop})
        cols_amount = [
            "amount_4917619A_1",
        ]
        expr_list = []
        for col in cols_amount:
            expr_list.append((pl.col(col)/pl.col("credamount_770A_1_static0")).alias(f"{col}1"))
            expr_list.append((pl.col(col)/pl.col("maininc_215A_2_static0")).alias(f"{col}2"))
        df = df.with_columns(expr_list)
        df = df.sort("deductiondate_4917603D_1")
        return df
    
    def _process_taxregistryc1(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Process taxregistry data from source c.

        Arguments:
            df: polars.DataFrame with data.
        Returns:
            df: polars.DataFrame with processed data.
        """
        cols_drop = [
            "employername_160M"
        ]
        df = df.rename({col: f"{col}_1" for col in df.columns if col not in self.cols_keep+cols_drop})
        cols_amount = [
            "pmtamount_36A_1",
        ]
        expr_list = []
        for col in cols_amount:
            expr_list.append((pl.col(col)/pl.col("credamount_770A_1_static0")).alias(f"{col}1"))
            expr_list.append((pl.col(col)/pl.col("maininc_215A_2_static0")).alias(f"{col}2"))
        df = df.with_columns(expr_list)
        df = df.sort("processingdate_168D_1")
        return df

    def _process_aggregations_all(self, df: pl.DataFrame, cols: list | None = None, suffix: str = "") -> pl.DataFrame:
        """
        Process aggregations for the dataset.

        Arguments:
            df: polars.DataFrame with data.
            cols: A list of columns to process.
            suffix: A suffix to add to the column names.
        Returns:
            df: polars.DataFrame with processed data.
        """
        if cols is None:
            cols = df.columns
        features = {i: "_".join(i.split("_")[:-1]) for i in cols if i not in self.cols_keep}
        expr_list = []
        for col, col_original in features.items():
            expr_list.extend(self._process_aggregations(col, self.features_df.loc[col_original], suffix=suffix))
        expr_list.append(pl.col(COL_ID).count().alias(f"{COL_ID}_1_count{suffix}"))
        if self.name in ["creditbureau1", "creditbureau2"]:
            expr_list.append(pl.lit(self.source).alias(f"source_1{suffix}"))
        df = df.group_by(COL_ID).agg(expr_list).sort(COL_ID)
        return df

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
        expr_list = [i for i in expr_list if i is not None]
        return expr_list


class BaseDataLoader:
    """
    A class to load the base data for the model.
    """
    def __init__(self, test, df_props):
        """
        Initialize the BaseDataLoader.
        """
        self.test = test
        self.df_props = df_props

    def load_data(self, case_ids: list | None = None) -> pl.DataFrame:
        """
        Load data based on mode.
        Arguments:
            test: A flag to indicate if the data should be loaded in test mode.
            case_ids: A list of case IDs to filter the data.
        Returns:
            df: polars.DataFrame with base data
        """
        batch_reader = BatchReader(self.df_props, self.test)
        dfs = []
        for  df, source in batch_reader.batches(case_ids):
            dfs.append(df)
        df = pl.concat(dfs, how="vertical_relaxed")

        if case_ids is not None:
            df = df.filter(pl.col(COL_ID).is_in(case_ids))
        if self.test:
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
        for suf in ["", "_1month"]:
            df = df.with_columns(
                (pl.sum_horizontal([
                    f"amount_4527230A_1_mean{suf}_taxregistrya1",
                    f"amount_4917619A_1_mean{suf}_taxregistryb1",
                    f"pmtamount_36A_1_mean{suf}_taxregistryc1"
                ])/3).alias(f"amount_1_mean{suf}_taxregistry1"),
                pl.min_horizontal([
                    f"amount_4527230A_1_min{suf}_taxregistrya1",
                    f"amount_4917619A_1_min{suf}_taxregistryb1",
                    f"pmtamount_36A_1_min{suf}_taxregistryc1"
                ]).alias(f"amount_1_min{suf}_taxregistry1"),
                pl.max_horizontal([
                    f"amount_4527230A_1_max{suf}_taxregistrya1",
                    f"amount_4917619A_1_max{suf}_taxregistryb1",
                    f"pmtamount_36A_1_max{suf}_taxregistryc1"
                ]).alias(f"amount_1_max{suf}_taxregistry1"),
                pl.coalesce(pl.col([
                    f"amount_4527230A_1_last{suf}_taxregistrya1",
                    f"amount_4917619A_1_last{suf}_taxregistryb1",
                    f"pmtamount_36A_1_last{suf}_taxregistryc1"
                ])).alias(f"amount_1_last{suf}_taxregistry1"),
            )
            
        # Coalesce birthdate columns
        df = df.with_columns(
            pl.coalesce(
                pl.col("birthdate_574D_2_staticcb0"), 
                pl.col("birth_259D_1_person1"),
            ).alias("birthdate_1_final0")
        )
        df = df.with_columns(
            pl.coalesce(pl.col("employedfrom_700D_1_applprev1"), pl.col("empl_employedfrom_271D_1_person1")).alias("employed_length_1_final0")
        )
        df = df.with_columns(
            pl.coalesce(pl.col("maininc_215A_1_static0"), pl.col("mainoccupationinc_384A_1_person1")).alias("mainoccupationinc_1_final0")
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
    

class DataProcessor4:
    """
    A utility class for processing data.
    It loads, processes, and prepares datasets for training and testing,
    handling feature engineering, aggregation, and encoding as specified by the dataset's properties.
    """
    def __init__(self, name) -> None:
        """
        Initializes the DataProcessor with a specific name and loads the dataset properties and features information.

        Arguments:
            name: A unique name for the data Utils instance, used for identifying processed datasets.
        """
        self.name = name
        self.path_data = f"{PATH_DATA_PROC}/{self.name}"
        if not KAGGLE:
            utils.create_folder(self.path_data)

        # Dict with dataframes to process
        with open(os.path.join(PATH_FEATURES, f"dfs_props_{VERSION}.pkl"), 'rb') as handle:
            self.dfs_props = pickle.load(handle)

        # Dataframe with features information
        self.features_df = utils.get_features_df(self.dfs_props, VERSION)

    def get_data(self, test: bool = False, fit: bool = False, load: bool = False, case_ids = None, save_to_disc: bool = True) -> pd.DataFrame:
        """
        Retrieves and processes the dataset based on specified conditions and configurations.

        Arguments:
            test: Specifies whether to load test data.
            fit: Specifies whether to fit the transformer.
            load: Specifies whether to load pre-processed data from disk.
            case_ids: A list of case IDs to filter the data. If None, no filtering is applied.
            save_to_disc: Specifies whether to save the processed data to disk.
        Returns:
            pd.DataFrame: A DataFrame containing processed features and target variables.
        """
        if load:
            df = self.load_data()
            df = utils.reduce_mem_usage(df)
            print(f"Number of columns: {len(df.columns)}")
            return df

        # Get base data
        base_data_loader = BaseDataLoader(test, self.dfs_props["base"])
        df = base_data_loader.load_data(case_ids)
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
        df = utils.reduce_mem_usage(df)

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
    