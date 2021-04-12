import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

import komono.pandas._reduce_memory as rd


@pytest.fixture
def base_data():
    return {
        "int8": [-128, 127],
        "int16": [-129, 127],
        "Int8": [None, 127],
        "Str": ["foo", "bar"],
    }


@pytest.fixture
def base_dtype():
    return {
        "int8": "int64",
        "int16": "int64",
        "Int8": "Int64",
        "Str": "string",
    }


@pytest.fixture
def base_dataframe(base_data, base_dtype) -> pd.DataFrame:
    return pd.DataFrame.from_dict(base_data).astype(base_dtype)


@pytest.mark.parametrize(
    "min_,max_,expected_dtype",
    [
        (-128, 127, "int8"),
        (-128, 128, "int16"),
        (-129, 127, "int16"),
        (-129, 128, "int16"),
        (-32_768, 32_767, "int16"),
        (-32_768, 32_768, "int32"),
        (-32_769, 32_767, "int32"),
        (-32_769, 32_768, "int32"),
        (-2_147_483_648, 2_147_483_647, "int32"),
        (-2_147_483_648, 2_147_483_648, "int64"),
        (-2_147_483_649, 2_147_483_647, "int64"),
        (-2_147_483_649, 2_147_483_648, "int64"),
    ],
)
def test_reduce_integer_series_not_nullable(min_, max_, expected_dtype):
    series = pd.Series([min_, max_], dtype="int64")
    dtype = str(series.dtype)
    expected = pd.Series([min_, max_], dtype=expected_dtype)
    actual = rd._reduce_integer_series(series, dtype=dtype)
    assert_series_equal(actual, expected)


@pytest.mark.parametrize(
    "min_,mid,max_,expected_dtype",
    [
        (-128, None, 127, "Int8"),
        (-128, None, 128, "Int16"),
        (-129, None, 127, "Int16"),
        (-129, None, 128, "Int16"),
        (-32_768, None, 32_767, "Int16"),
        (-32_768, None, 32_768, "Int32"),
        (-32_769, None, 32_767, "Int32"),
        (-32_769, None, 32_768, "Int32"),
        (-2_147_483_648, None, 2_147_483_647, "Int32"),
        (-2_147_483_648, None, 2_147_483_648, "Int64"),
        (-2_147_483_649, None, 2_147_483_647, "Int64"),
        (-2_147_483_649, None, 2_147_483_648, "Int64"),
    ],
)
def test_reduce_integer_series_nullable(min_, mid, max_, expected_dtype):
    series = pd.Series([min_, mid, max_], dtype="Int64")
    dtype = str(series.dtype)
    expected = pd.Series([min_, mid, max_], dtype=expected_dtype)
    actual = rd._reduce_integer_series(series, dtype=dtype)
    assert_series_equal(actual, expected)


@pytest.mark.parametrize(
    "min_,max_,expected_dtype",
    [
        (-65500.0, 65500.0, "float16"),
        (-65500.0, 65600.0, "float32"),
        (-65600.0, 65500.0, "float32"),
        (-65600.0, 65600.0, "float32"),
        (-3.4028e38, 3.4028e38, "float32"),
        (-3.4028235e38, 3.4028335e38, "float64"),
        (-3.4028335e38, 3.4028235e38, "float64"),
        (-3.4028335e38, 3.4028335e38, "float64"),
    ],
)
def test_reduce_float_series(min_, max_, expected_dtype):
    series = pd.Series([min_, max_], dtype="float64")
    expected = pd.Series([min_, max_], dtype=expected_dtype)
    actual = rd._reduce_float_series(series)
    assert_series_equal(actual, expected)


def test_reduce_memory_usage(base_data, base_dataframe):
    expected = pd.DataFrame.from_dict(data=base_data,).astype(
        {
            "int8": "int8",
            "int16": "int16",
            "Int8": "Int8",
            "Str": "string",
        }
    )
    actual = rd.reduce_memory_usage(base_dataframe, verbose=True)
    assert_frame_equal(actual, expected)
