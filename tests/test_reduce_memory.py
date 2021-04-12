import pandas as pd
import pytest
from pandas.testing import assert_series_equal

import komono.pandas._reduce_memory as rd


@pytest.fixture
def int8_series() -> pd.Series:
    data = [-128, 1, 2, 3, 127]
    return pd.Series(data, dtype="int64")


@pytest.fixture
def int16_series() -> pd.Series:
    data = [-32_768, -128, 1, 2, 3, 127, 32_767]
    return pd.Series(data, dtype="int64")


@pytest.fixture
def int32_series() -> pd.Series:
    data = [-2_147_483_648, -32_768, -128, 1, 2, 3, 127, 32_767, 2_147_483_647]
    return pd.Series(data, dtype="int64")


@pytest.fixture
def int64_series() -> pd.Series:
    data = [-9_223_372_036_854_775_808, 1, 2, 3, 9_223_372_036_854_775_807]
    return pd.Series(data, dtype="int64")


def test_int64_to_int8(int8_series):
    dtype = str(int8_series.dtype)
    expected = pd.Series([-128, 1, 2, 3, 127], dtype="int8")
    actual = rd._reduce_integer_series(int8_series, dtype=dtype)
    assert_series_equal(actual, expected)


def test_int64_to_int16(int16_series):
    dtype = str(int16_series.dtype)
    expected = pd.Series([-32_768, -128, 1, 2, 3, 127, 32_767], dtype="int16")
    actual = rd._reduce_integer_series(int16_series, dtype=dtype)
    assert_series_equal(actual, expected)


def test_int64_to_int32(int32_series):
    dtype = str(int32_series.dtype)
    expected = pd.Series(
        [-2_147_483_648, -32_768, -128, 1, 2, 3, 127, 32_767, 2_147_483_647],
        dtype="int32",
    )
    actual = rd._reduce_integer_series(int32_series, dtype=dtype)
    assert_series_equal(actual, expected)


def test_int64_to_int64(int64_series):
    dtype = str(int64_series.dtype)
    expected = pd.Series(
        [-9_223_372_036_854_775_808, 1, 2, 3, 9_223_372_036_854_775_807], dtype="int64"
    )
    actual = rd._reduce_integer_series(int64_series, dtype=dtype)
    assert_series_equal(actual, expected)
