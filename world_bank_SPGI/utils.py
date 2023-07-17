import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from pmdarima import auto_arima
from prophet import Prophet

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)


def provide_data(series_code: str, country_code: str) -> pd.DataFrame:
    """
    Downloads time series data from world bank based on given series and country code
    """
    url_api_schema = f"https://api.worldbank.org/v2/country/{country_code}/indicator/{series_code}?format=json"
    r = requests.get(url_api_schema)
    json_data = r.json()
    data = pd.json_normalize(json_data[1])

    # not using errors='coerce' is my choice, but can be added with issues tracker
    # i.e. by [date for date, converted in zip(df_before.data, df_after.data) if pd.isna(converted)]
    data.date = pd.to_datetime(data.date)
    data.value = pd.to_numeric(data.value)

    # Q: to implement - what if we have more than one value per year?
    # Q: to implement - what if we are missing data - how to deal with it?
    data = data[["date", "value"]].dropna()
    data = data.set_index("date")

    # Q: based on years gap I would talk and agree with you about how to set different start year of forecasts
    # For AFG we might use data from 2002-01-01
    # I used ffil for missing data as prophet gave negative numbers for some years.
    # Q: or we could transform prophet data by fitting log for model and convert forecast to normal scale
    df_resampled = data.resample("AS").asfreq()

    # quick check
    logging.info(
        f"Filled rows: {len(df_resampled) - len(data)}\nprimary number of rows:{len(data)}"
    )

    df_resampled["source"] = np.where(
        df_resampled.index.isin(data.index), "World Bank", "Resampled"
    )
    data = df_resampled.ffill()
    logging.info("World Bank data processed.")
    return data.sort_index()


def prediction_periods(
    dataframe: pd.DataFrame, model_name: str, cut_off_year: int = 2030
) -> int:
    """Provides number of years between cut_off_date and newest date in model dataframe"""
    max_series_year = (
        dataframe.index.max().year if model_name == "arima" else dataframe.ds.max().year
    )

    current_year = datetime.now().year

    if max_series_year > current_year:
        raise ValueError(
            f"Maximum year in data series ({max_series_year}) should not exceed the current year ({current_year})."
        )

    if max_series_year > cut_off_year:
        raise ValueError(
            f"The specified prediction year cut-off ({cut_off_year}) must be later than the last year of available data ({max_series_year})."
        )

    periods_number = cut_off_year - max_series_year
    return periods_number


def forecast_arima(df: pd.DataFrame) -> pd.DataFrame:
    # decided to have auto_arima trace
    model = auto_arima(df["value"], trace=True, error_action="warn")
    forecast_values = model.predict(
        n_periods=prediction_periods(df, model_name="arima")
    )
    forecast_df = pd.DataFrame({"arima_values": forecast_values})
    logging.info("Arima forecast done.")
    return forecast_df


def forecast_prophet(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index().rename(columns={"date": "ds", "value": "y"})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(
        periods=prediction_periods(df, model_name="prophet"),
        freq="Y",
        include_history=False,
    )
    forecast = model.predict(future)
    forecast_df = pd.DataFrame(
        {
            "date": forecast["ds"] + pd.Timedelta(days=1),
            "prophet_value": pd.to_numeric(forecast["yhat"]),
        }
    ).set_index("date")
    logging.info("Prophet forecast done.")
    return forecast_df


def concat_data(
    wb_df: pd.DataFrame, arima_df: pd.DataFrame, prophet_df: pd.DataFrame
) -> pd.DataFrame:
    forecast_df = (
        pd.concat([prophet_df, arima_df], axis=1)
        .assign(
            value=lambda df: (df["prophet_value"] + df["arima_values"]) / 2.0,
            source="Forecast",
        )
        .drop(["prophet_value", "arima_values"], axis=1)
    )

    union_df = pd.concat([wb_df, forecast_df], axis=0)
    union_df.index = union_df.index.strftime("%Y-%m-%d")
    logging.info("Data merged.")
    return union_df


def count_source_rows(df: pd.DataFrame, source_name: str) -> str:
    return str(df[df.source == source_name].shape[0])


def create_direction() -> Path:
    path = Path(__file__).parents[1] / "_output"
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_results(df: pd.DataFrame) -> None:
    result = {
        "world_bank_data_rows": count_source_rows(df, "World Bank"),
        "resample_data_rows": count_source_rows(df, "Resampled"),
        "forecast_data_rows": count_source_rows(df, "Forecast"),
        "historical_data_end": df[df.source == "World Bank"].index.max(),
        "forecast_start": df[df.source == "Forecast"].index.min(),
        "data": df.to_dict(orient="index"),
    }

    path = create_direction()
    df.to_csv(path / "data.csv", index_label="date")
    with open(path / "forecast.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)
    logging.info("Data saved and ready to view.")
