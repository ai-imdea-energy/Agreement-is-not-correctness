# Data

All files needed to run the code are included in this folder.

## Spain: `energy_dataset.csv`, `weather_features.csv`

Hourly data for Spain, 2015 to 2018, from the Kaggle dataset "Hourly energy demand generation and weather" by Nicholas Jhana:
https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather

The files are included unchanged. According to the dataset description, demand and generation come from the ENTSO-E Transparency Platform, prices from Red Eléctrica de España, and the weather for Madrid, Barcelona, Valencia, Seville and Bilbao from the OpenWeather API. All cleaning is done inside the code.

## Portugal: `portugal/energy_dataset.csv`, `portugal/weather_features.csv`

Built by Part 5 of the script from two public sources and written in the same format as the Spanish files:

* load, the operator's day-ahead forecast, and solar and wind generation: Open Power System Data, time series package, version 2020-10-06 (https://data.open-power-system-data.org/time_series/2020-10-06/)
* hourly weather for Lisbon, Porto, Braga, Coimbra and Faro: Open-Meteo historical weather archive (https://open-meteo.com/)

If these two files are deleted, Part 5 downloads the raw data again (about 135 MB, into `portugal/raw/`) and rebuilds them.

## Synthetic benchmark

No files are needed. The benchmark is generated inside the code from fixed seeds.

Please credit the original sources above when you reuse the data.
