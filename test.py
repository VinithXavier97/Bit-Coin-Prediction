# -*- coding: utf-8 -*-

import pandas as pd
import datetime
df = pd.read_csv("bitcoin_ticker.csv")

print(df.shape)

data = pd.read_csv("bitstampUSD_1-min_data_2012-01-01_to_2019-08-12.csv")

print(data.shape)

data = data.dropna()

data['date'] = pd.to_datetime(data['Timestamp'],unit='s').dt.date

print(data.shape)

print(data)

dates= df['date_id'].unique()

data_by_date = df[['low','high','volume']][df['date_id'] == "2017-05-31"]

print(data_by_date)

print(dates)