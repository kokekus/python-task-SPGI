from datetime import datetime

import pandas as pd
import pytest

from utils import count_source_rows, prediction_periods, provide_data


@pytest.mark.parametrize(
    "series_code, country_code", [("SP.POP.TOTL", "AFG"), ("NY.GDP.MKTP.CD", "USA")]
)
def test_provide_data(series_code, country_code):
    data = provide_data(series_code, country_code)
    assert isinstance(data, pd.DataFrame)
    assert not data.empty


@pytest.mark.parametrize(
    "dataframe, model_name, cut_off_year, expected_periods",
    [
        (
            pd.DataFrame(
                {"value": [10, 20, 30]},
                index=[
                    datetime(2010, 1, 1),
                    datetime(2011, 1, 1),
                    datetime(2013, 1, 1),
                ],
            ),
            "arima",
            2035,
            22,
        ),
        (
            pd.DataFrame(
                {
                    "ds": [
                        datetime(2020, 1, 1),
                        datetime(2021, 1, 1),
                        datetime(2022, 1, 1),
                    ]
                }
            ),
            "prophet",
            2030,
            8,
        ),
        (
            pd.DataFrame(
                {
                    "ds": [
                        datetime(2000, 1, 1),
                        datetime(2005, 1, 1),
                        datetime(2010, 1, 1),
                    ]
                }
            ),
            "prophet",
            2025,
            15,
        ),
    ],
)
def test_prediction_periods(dataframe, model_name, cut_off_year, expected_periods):
    periods = prediction_periods(dataframe, model_name, cut_off_year)
    assert periods == expected_periods


@pytest.mark.parametrize(
    "df, source_name, expected",
    [
        (
            pd.DataFrame({"source": ["World Bank", "World Bank", "Forecast"]}),
            "World Bank",
            "2",
        ),
        (
            pd.DataFrame({"source": ["Resampled", "Resampled", "Resampled"]}),
            "Resampled",
            "3",
        ),
        (pd.DataFrame({"source": ["Forecast"]}), "Forecast", "1"),
        (pd.DataFrame({"source": ["World Bank", "Resampled"]}), "Forecast", "0"),
    ],
)
def test_count_source_rows(df, source_name, expected):
    assert count_source_rows(df, source_name) == expected
