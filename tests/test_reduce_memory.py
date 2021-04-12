import pandas as pd
import pytest
from pandas.testing import assert_series_equal

import komono.pandas._reduce_memory as rd


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
