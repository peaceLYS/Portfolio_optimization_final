'''
This is Black-Litterman model created by Abhi and edited by YishiLiu
'''
from numpy import matrix, array, zeros, empty, sqrt, ones, dot,mean,cov, transpose, linspace
from numpy.linalg import inv, pinv
import pylab as plt
import pandas as pd
import pandas_datareader as pdr
from pandas import Series, DataFrame
#from structures.quote import QuoteSeries
import datetime
import scipy.optimize
import random
import DataPreparation_BL
from plotly.tools import mpl_to_plotly
import numpy as np
# load data_out function that takes list of tickers and dates as input and returns list of tickers, daily prices and market caps
def load_data_net(symbolsList, start, end, cap):
        all_data = {}
        for symbol in symbolsList:
            all_data[symbol] = pdr.get_data_yahoo(symbol, start, end)
        price = DataFrame({tic: data['Adj Close']
                            for tic, data in all_data.items()})
        #drop all rows that have any NaN values
        price.dropna(axis=0,how='any',inplace=True)
        i = 0
        prices_out, caps_out = [], []
        for symbol in symbolsList:
            prices_out.append(price.iloc[:,i])
            caps_out.append(cap[symbol])
            i = i + 1
        return symbolsList, prices_out, caps_out

# Function takes historical stock prices together with market capitalizations and calculates
# names       - array of assets' names
# prices      - array of historical (daily) prices
# caps        - array of assets' market capitalizations
# returns:
# names       - array of assets' names
# weights     - array of assets' weights (derived from mkt caps)
# expreturns  - expected returns based on historical data
# covars          - covariance matrix between assets based on historical data
def assets_meanvar(names, prices, caps):
        prices = matrix(prices)                         # create numpy matrix from prices
        weights = array(caps) / sum(caps)       # create weights

        # create matrix of historical returns
        rows, cols = prices.shape
        returns = empty([rows, cols-1])
        for r in range(rows):
                for c in range(cols-1):
                        p0, p1 = prices[r,c], prices[r,c+1]
                        returns[r,c] = (p1/p0)-1

        # calculate expected returns
        expreturns = []
        for r in range(rows):
                expreturns.append(mean(returns[r]))
                
        # calculate covariances
        covars = cov(returns)
        print("!!!!",expreturns)
        expreturns=np.array(expreturns)
        expreturns = (1+expreturns)**250-1      # Annualize expected returns
        print("!!!!!!!!!!",expreturns)
        #expreturns=list(expreturns)
        covars = covars * 250                           # Annualize covariances

        return names, weights, expreturns, covars

#       rf              risk free rate
#       lmb             lambda - risk aversion coefficient
#       C               assets covariance matrix
#       V               assets variances (diagonal in covariance matrix)
#       W               assets weights
#       R               assets returns
#       mean    portfolio historical return
#       var             portfolio historical variance
#       Pi              portfolio equilibrium excess returns
#       tau     scaling factor for Black-litterman

# Calculates portfolio mean return
def portmean(W, R):
        return sum(R*W)

# Calculates portfolio variance of returns
def portvar(W, C):
        return dot(dot(W, C), W)

# Combination of the two functions above - mean and variance of returns calculation
def portmeanvar(W, R, C):
        return portmean(W, R), portvar(W, C)

# Given risk-free rate, assets returns and covariances, this function calculates
# mean-variance frontier and returns its [x,y] points in two arrays
def solve_frontier(R, C, rf):
        def fitness(W, R, C, r):
                # For given level of return r, find weights which minimizes
                # portfolio variance.
                mean, var = portmeanvar(W, R, C)
                # Big penalty for not meeting stated portfolio return effectively serves as optimization constraint
                penalty = 50*abs(mean-r)
                return var + penalty
        frontier_mean, frontier_var, frontier_weights = [], [], []
        n = len(R)      # Number of assets in the portfolio
        for r in linspace(min(R), max(R), num=20): # Iterate through the range of returns on Y axis
                W = ones([n])/n         # start optimization with equal weights
                b_ = [(0,1) for i in range(n)]
                c_ = ({'type':'eq', 'fun': lambda W: sum(W)-1. })
                optimized = scipy.optimize.minimize(fitness, W, (R, C, r), method='SLSQP', constraints=c_, bounds=b_)
                if not optimized.success:
                        raise BaseException(optimized.message)
                # add point to the min-var frontier [x,y] = [optimized.x, r]
                frontier_mean.append(r)                                                 # return
                frontier_var.append(portvar(optimized.x, C))   # min-variance based on optimized weights
                frontier_weights.append(optimized.x)
        return array(frontier_mean), array(frontier_var), frontier_weights

# Given risk-free rate, assets returns and covariances, this
# function calculates weights of tangency portfolio with respect to
# sharpe ratio maximization
def solve_weights(R, C, rf):
        def fitness(W, R, C, rf):
                mean, var = portmeanvar(W, R, C)      # calculate mean/variance of the portfolio
                util = (mean - rf) / sqrt(var)          # utility = Sharpe ratio
                return 1/util                                           # maximize the utility, minimize its inverse value
        n = len(R)
        W = ones([n])/n                                         # start optimization with equal weights
        b_ = [(0.,1.) for i in range(n)]        # weights for boundaries between 0%..100%. No leverage, no shorting
        c_ = ({'type':'eq', 'fun': lambda W: sum(W)-1. })       # Sum of weights must be 100%
        optimized = scipy.optimize.minimize(fitness, W, (R, C, rf), method='SLSQP', constraints=c_, bounds=b_)
        if not optimized.success:
                raise BaseException(optimized.message)
        return optimized.x

def print_assets(names, W, R, C):
        name=[]
        weight=[]
        ret=[]
        dev=[]
        corre=[[] for i in range(len(names))]
    
        #print("%-10s %6s %6s %6s %s" % ("Name", "Weight", "Return", "Dev", "   Correlations"))
        for i in range(len(names)):
                #print("%-10s %5.1f%% %5.1f%% %5.1f%%    " % (names[i], 100*W[i], 100*R[i], 100*C[i,i]**.5), end='')
                name.append(names[i])
                weight.append(round(100*W[i],2))
                ret.append(round(100*R[i],2))
                dev.append(round(100*C[i,i]**.5,2))
                for j in range(i+1):
                        corr = C[i,j] / (sqrt(C[i,i]) * (sqrt(C[j,j]))) # calculate correlation from covariance
                        corre[i].append(round(corr,2))
                        #print("%.3f " % corr, end='')
                print()
        df=pd.DataFrame({'Name':name,'Weight':weight,'Return':ret,'Dev':dev})
        corre=pd.DataFrame(corre)
        corre=corre.fillna(0)
        return df,corre
        print(df)
        print(corre)
def optimize_and_display(names, R, C, rf):
        # optimize
        W = solve_weights(R, C, rf)
        mean, var = portmeanvar(W, R, C)                                              # calculate tangency portfolio
        f_mean, f_var, f_weights = solve_frontier(R, C, rf) 
        df,corr=print_assets(names, W, R, C)
        return mean,var,f_mean,f_var, f_weights,df,corr
        '''
        n = len(names)
        plt.scatter([C[i,i]**.5 for i in range(n)], R, marker='x',color=color)  # draw assets
        for i in range(n):                                                                              # draw labels
                plt.text(C[i,i]**.5, R[i], '  %s'%names[i], verticalalignment='center', color=color)
        plt.scatter(var**.5, mean, marker='o', color=color)                 # draw tangency portfolio
        plt.plot(f_var**.5, f_mean, color=color)                                    # draw min-var frontier
        plt.xlabel('$\sigma$'), plt.ylabel('$r$')
        plt.grid(True)
#       show()
        '''
        # Display weights
        #m = empty([n, len(f_weights)])
        #for i in range(n):
        #       for j in range(m.shape[1]):
        #               m[i,j] = f_weights[j][i]
        #stackplot(f_mean, m)
        #show()

# given the pairs of assets, prepare the views and link matrices. This function is created just for users' convenience
def prepare_views_and_link_matrix(names, views):
        r, c = len(views), len(names)
        Q = [views[i][3] for i in range(r)]     # view matrix
        P = zeros([r, c])                                       # link matrix
        nameToIndex = dict()
        for i, n in enumerate(names):
                nameToIndex[n] = i
        for i, v in enumerate(views):
                name1, name2 = views[i][0], views[i][2]
                P[i, nameToIndex[name1]] = +1 if views[i][1]=='>' else -1
                P[i, nameToIndex[name2]] = -1 if views[i][1]=='>' else +1
        return array(Q), P