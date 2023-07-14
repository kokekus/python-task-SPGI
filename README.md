# python-task-SPGI

Given a World Bank time series string and country, script will:
1.	Download that series and country from the World Bank 
2.	Clean it appropriately.
3.	Apply an average ensemble (of two forecasts methods) to the time series.
4.	Store the full time series to disk, including the forecast, in a csv and json file format.

## Forecasts methods
* https://pypi.org/project/pmdarima/ is one Python implementation of ARIMA 
* https://pypi.org/project/fbprophet/ is the Prophet package for Python

### To get you started
Script params are taken from  `config.yaml` Following parameters are needed:
* `series_code` - world bank time series code, i.e.'NY.GDP.MKTP.CN'
* `country_code` - country code, i.e. 'afg'

Function that uses the World Bank API to download the data, as per this API documentation https://datahelpdesk.worldbank.org/knowledgebase/articles/898581. 
The query string for that example is https://api.worldbank.org/v2/country/afg/indicator/NY.GDP.MKTP.CN

How to run:
```bash
python main.py 
```

### Warning
I and other people have reported problems installing Prophet and the pystan dependencies. After long struggle I manage to complete instalation. I suggest downgrade your python version to 3.7 before start.

## Flow
Description on the high level:
1. Install requirements `requirements.txt`
2. Configure `config.yaml` if needed
3. Run script `python main.py`
4. Summary in csv and json get saved to the `_output` folder.

## Output

`_output/data.csv` - script output

`_output/forecast.json` - data provided in data.csv in json format with additional insights