import sys
from typing import List

import numpy as np
import pandas as pd

NUMERICAL_DTYPES = {
    "int8",
    "int16",
    "int32",
    "int64",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "float8",
    "float16",
    "float32",
    "float64",
}


def reduce_memory_usage(dataframe: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    # The implementation of this function is based on
    # https://github.com/TeruakiUeda/faster_reduce_mem_usage
    tmp: List[pd.Series] = []
    raw_memory_usage = dataframe.memory_usage().sum() / 2.0 ** 20
    for col in dataframe.columns:
        series = dataframe[col]
        dtype = str(series.dtype)
        if dtype in NUMERICAL_DTYPES:
            if dtype.lower().startswith("int"):
                tmp.append(_reduce_integer_series(series, dtype))
            else:
                tmp.append(_reduce_float_series(series))
        else:
            tmp.append(series)
    ret: pd.DataFrame = pd.concat(tmp, axis="columns")
    if verbose:
        reduced_memory_usage = ret.memory_usage().sum() / 2.0 ** 20
        reduction_ratio = (raw_memory_usage - reduced_memory_usage) / raw_memory_usage
        print(
            f"Memory usage decreased to {reduced_memory_usage:.3g}MiB: "
            f"{reduction_ratio:.2%} reduction",
            file=sys.stderr,
        )
    return ret


def _reduce_integer_series(series: pd.Series, dtype: str) -> pd.Series:
    is_nullable = dtype.startswith("Int")
    max_value = series.max()
    min_value = series.min()
    if np.iinfo(np.int8).min <= min_value and max_value <= np.iinfo(np.int8).max:
        if is_nullable:
            return series.astype("Int8")
        return series.astype("int8")
    if np.iinfo(np.int16).min <= min_value and max_value <= np.iinfo(np.int16).max:
        if is_nullable:
            return series.astype("Int16")
        return series.astype("int16")
    if np.iinfo(np.int32).min <= min_value and max_value <= np.iinfo(np.int32).max:
        if is_nullable:
            return series.astype("Int32")
        return series.astype("int32")
    return series


def _reduce_float_series(series: pd.Series) -> pd.Series:
    max_value = series.max()
    min_value = series.min()
    if np.finfo(np.float16).min <= min_value and max_value <= np.finfo(np.float16).max:
        return series.astype("float16")
    if np.finfo(np.float32).min <= min_value and max_value <= np.finfo(np.float32).max:
        return series.astype("float32")
    return series
