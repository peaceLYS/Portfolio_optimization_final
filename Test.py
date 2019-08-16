import pandas as pd
from pandas import DataFrame
import numpy as np
import math
from pypfopt.efficient_frontier import EfficientFrontier
import DataPreparation as DP
df = DP.ReadFromFile('test.csv')
price = DP.Yahoo(df,2017,2018)
(mu,S,returns) = DP.GetFromPrice(price)
ef = EfficientFrontier(mu, S, weight_bounds=((0, 1),
(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1)))
print(type(returns))
dates = returns.index.tolist()
weights=ef.max_sharpe()
weight = list(weights.values())

weight2 = [ float('%.2f' % elem) for elem in weight ] #round results

pf_returns = returns.mul(weight2).sum(axis = 1).div(0.01)
pf_returns = [ '%.2f' % elem for elem in pf_returns ]
dic = {'Dates':dates,'Portfolio Returns':pf_returns}
df2 = pd.DataFrame(dic)
print(df2)

'''
df = pd.DataFrame({'angles': [1, 3, 4],
                   'degrees': [2, 3, 4]},
                  index=['circle', 'triangle', 'rectangle'])

print(df)

list1 = [2,2]
#list2= np.array(list1).reshape((-1,1))

#list2 = list1.reshape(len(list1),1)
#print(math.matrix(list1).T)
result = df.mul(list1)
print(result)
'''
