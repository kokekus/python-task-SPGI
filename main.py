from world_bank_SPGI.config_reader import read_config
from world_bank_SPGI.utils import (
    concat_data,
    forecast_arima,
    forecast_prophet,
    provide_data,
    save_results,
)


def main():
    config = read_config()

    data = provide_data(
        series_code=config["series_code"], country_code=config["country_code"]
    )
    arima_forecast = forecast_arima(data)
    prophet_forecast = forecast_prophet(data)

    merged_data = concat_data(
        wb_df=data, arima_df=arima_forecast, prophet_df=prophet_forecast
    )

    save_results(merged_data)


if __name__ == "__main__":
    main()
