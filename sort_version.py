# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 23:17:50 2019

@author: liuyishi
"""

import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import math
import pandas_datareader as pdr
import datetime
pd.core.common.is_list_like = pd.api.types.is_list_like
import yfinance as yf
from scipy.optimize import minimize
#yf.pdr_override()

#***** pip install PyPortfolioOpt ******
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
#from pypfopt.hierarchical_risk_parity import hrp_portfolio
from pypfopt.value_at_risk import CVAROpt
from pypfopt.hierarchical_risk_parity import HRPOpt
from empyrical import downside_risk,sortino_ratio,calmar_ratio,omega_ratio,tail_ratio,max_drawdown
#from pypfopt import discrete_allocation
# Reading in the data; 
fileName = 'test.csv'
start = datetime.date(2000, 1, 1)#resolve if only input year later
end = datetime.date(2019, 2, 27)
if fileName[fileName.rfind(".")+1:] == 'csv':
     df = pd.read_csv(fileName)
elif fileName[fileName.rfind(".")+1:] == 'xlsx':
     df = pd.read_excel(fileName)
else : 
    print('Wrong File Type')

df.dropna(axis=0,how='any') #drop all rows that have any NaN values
Names = df.iloc[:,0] #first column

tickerList = Names.array #convert Series to array

#retrieve data from yahoo finance
all_data = {}
for ticker in tickerList:
    all_data[ticker] = pdr.get_data_yahoo(ticker, start, end)

#store data in DataFrame
price = DataFrame({tic: data['Adj Close']
                    for tic, data in all_data.items()})
#drop all rows that have any NaN values
price.dropna(axis=0,how='any',inplace=True) 
Benchmark=price.iloc[:,-1]
price=price.iloc[:,:-1]
#print(price)
import urllib
import requests
requests.packages.urllib3.disable_warnings()
from bs4 import BeautifulSoup
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context
url = "https://ycharts.com/indicators/1_month_treasury_rate"
headers = {}
headers['User-Agent'] = "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.3"
req = urllib.request.Request(url,headers = headers)
response = urllib.request.urlopen(req)
soup = BeautifulSoup(response)
#print(soup.prettify())
#extract rate from sourse code
s = soup.find("td",{'class': 'col2'}).string
s = s.replace(" ", "")
rf = float(s[0:5])/100
print(rf)


# User can choose to calculate annual return using monthly data or yearly data
def calreturn(price,period):
    first_m=price.apply(lambda x:x.resample(period).first())
    last_m=price.apply(lambda x:x.resample(period).last())
    returns=last_m/first_m-1
    return returns
    
def predata(price,yearly=False):
    if yearly:
        returns=calreturn(price,'Y')
        mu=returns.mean()
        
    else:
        returns=calreturn(price,'M')
        mu=returns.mean()*12
    S = risk_models.sample_cov(price)
    return mu,S
# we don't allow to short asset
mu,S=predata(price,yearly=False)
num_ticker=len(mu)

def H_RiskParity(price,period):
    returns=calreturn(price,'M')
    rp=HRPOpt(returns)
    #print(rp.hrp_portfolio())
    return rp.hrp_portfolio()

H_RiskParity(price,period='M')