'''This file contains five functions ReadFromFile, Yahoo, YahooMC, GetFromPrice, GetRF
   They are used to prepare necessary data for optimization
   
   Created by Abhi and edited by YishiLiu
'''
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import math
import pandas_datareader as pdr
import datetime
pd.core.common.is_list_like = pd.api.types.is_list_like
import yfinance as yf
yf.pdr_override()

import urllib
import requests
#requests.packages.urllib3.disable_warnings()
from bs4 import BeautifulSoup
import ssl

# Reading in the data;
#fileName = 'test.csv'
'''The required format of file: first column is tickers, second is current weights
   @param fileName: the name of file user uploaded
   @return df : a dataframe that contains content of the file
'''
def ReadFromFile(fileName):
    if fileName[fileName.rfind(".")+1:] == 'csv':
         df = pd.read_csv(fileName)
    elif fileName[fileName.rfind(".")+1:] == 'xlsx':
         df = pd.read_excel(fileName)
    else :
        return 'Wrong File Type'
    return df

'''Requires that the current allocations is the second column
@param df: df from csv
@return an dictionary of weights
'''
def GetAllo(df) :
    df.dropna(axis=0,how='any')
    Names = df.iloc[:,0].tolist()
    Allo = df.iloc[:,1].tolist()
    yourAlllo = {'Assets':Names, 'Provided Allocation':Allo}
    return yourAlllo

'''This function is used to getting price from yahoo yfinance
@param df: dataframe with tickers in column
       startY,endY,startM,endM : start & end years and month
       default value of month is 1
@return price: a dataframe contains price all over the required years
'''
def Yahoo(df, startY, endY, startM = 1, endM = 1):
    df.dropna(axis=0,how='any') #drop all rows that have any NaN values
    Names = df.iloc[:,0] #first column
    tickerList = Names.array #convert Series to array
    start = datetime.date(startY, startM, 1)#resolve if only input year later
    end = datetime.date(endY, endM, 1)
    #retrieve data from yahoo finance
    all_data = {}
    for ticker in tickerList:
        all_data[ticker] = pdr.get_data_yahoo(ticker, start, end)
    #store data in DataFrame
    price = DataFrame({tic: data['Adj Close']
                        for tic, data in all_data.items()})
    #drop all rows that have any NaN values, i.e. price is not availble
    price.dropna(axis=0,how='any',inplace=True)
    return price

'''This function is used to get the current Market Capitalization of specified Tickers,
which is used to calculate Benchmark portfolio for Black-litterman allocation
@param df : a dataframe, first column is tickers
@return MC: a dictionary, calculated Market Capitalization by volume * Price'''
def YahooMC(df):
    df.dropna(axis=0,how='any') #drop all rows that have any NaN values
    Names = df.iloc[:,0] #first column
    tickerList = Names.array #convert Series to array
    today = datetime.date.today()
    # get the last 7 days data because today might not be a trading day
    start = today - datetime.timedelta(days=7)
    #retrieve data from yahoo finance
    all_data = {}
    for ticker in tickerList:
        all_data[ticker] = pdr.get_data_yahoo(ticker, start, today)
    #store data in DataFrame
    price = DataFrame({tic: data['Adj Close']
                        for tic, data in all_data.items()})
    volume = DataFrame({tic: data['Volume']
                        for tic, data in all_data.items()})
    p = price.iloc[len(price)-1,:].to_dict() # get the last row and put into a dictionary
    v = volume.iloc[len(volume)-1,:].to_dict()
    MC = dict()
    MC['^GSPC'] = 14.90e12
    for k , va in p.items():
        MC[k] = va * v[k]
    return MC

'''This method is used to get return from daily price in a certain frequency
@param price: a dataframe contains price all over the certain years
       frequency : multiplier to annulize monthly data,
                   default 12 is for monthly return
@return mu, cov : µ and ∑ calculated from historical return
'''
def GetFromPrice(price, frequency = 12):
    #get monthly data from daily data
    P_end=price.apply(lambda x:x.resample('M').last())
    P_begin=price.apply(lambda x: x.resample('M').first())
    #preparing expected returns and a risk model
    returns=(P_end/P_begin-1).dropna(how="all")
    mu = returns.mean() * frequency
    cov = returns.cov() * frequency
    return mu, cov, returns

#made for annual return Tab
def GetAnnualR(price):
    P_end=price.apply(lambda x:x.resample('Y').last())
    P_begin=price.apply(lambda x: x.resample('Y').first())
    #preparing expected returns and a risk model
    returns=(P_end/P_begin-1).dropna(how="all")
    #mu = returns.mean() * frequency
    #cov = returns.cov() * frequency
    return returns

'''A simple function used to grab one month treatry bill rate from web using urllib
@return risk free rate, a float
'''
def GetRF():
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
    return rf
