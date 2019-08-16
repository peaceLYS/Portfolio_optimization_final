''' 
Created by Chaoyi Ye
'''
from Optimizer import All_frontier
from pypfopt.efficient_frontier import EfficientFrontier
import numpy as np
import scipy.stats as stat

''' This function use resampling method to get efficient frontier portfolio
# @param: mu, cov : expected return and covarive calculated directly from our his price data,
#                   they are the original mu and cov before resampling
# @param: rf: risk free rate
# return: averaged resampled mu and cov
'''
def Resampling_EF(mu, cov, weight_bounds, rf=0.02):
    ww =[]
    mm=[]
    ss=[]
    # Generate a distribution of returns
    return_Dis = stat.multivariate_normal(mu, cov)
    #Repeat 200 times
    for i in range(10):
        #draw 200 random samples, one sample includes n asset returns
        samples = return_Dis.rvs(200)
        # Estimate μ and Σ according to samples
        mu_est = samples.mean(0).reshape(len(mu), 1)#take mean along col, reshape it to N rows in 1 col
        cov_est = np.cov(samples.T)#T means transpose

        ef2 = EfficientFrontier(mu_est, cov_est, weight_bounds)
        #calculates weights for minimize volatility, max return, and some middle points for graphing
        #weights=list(ef.min_volatility().values())
        #w_all.append(weights)

        #Draw EfficientFrontier using old mu and cov
        (w, m, s)=All_frontier(mu,cov,ef2)
        #w2 = list(ef2.efficient_risk(risk_free_rate=rf,target_risk=0.250).values())
        #(m2, s2, sharpe2)=ef2.portfolio_performance(verbose=False,risk_free_rate=rf)

        ww.append(w)
        mm.append(m)
        ss.append(s)
    # calculate the average weights of 1000 times
    #w_average1 = np.mean(w_all, axis = 0)
    #w_average2 = np.mean(w_all2, axis = 0)
    w_average = np.mean(ww, axis = 0)
    mu_a = np.array(np.mean(mm,axis=0)).flatten()
    cov_a = np.mean(ss,axis = 0)
    #result = [w_average1, w_average2]#2 is max return, 1 is min vol
    return mu_a,cov_a
'''
# Reading in the data;
fileName = 'test.csv'
df = dp.ReadFromFile(fileName)
price = dp.Yahoo(df, 2000,2019)
(mu, cov, r) = dp.GetFromPrice(price)

rf = 0.02#dp.GetRF()
ef = EfficientFrontier(mu, cov, weight_bounds=((-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1)))
(w_f2,mu_f2, sigma_f2) = All_frontier(mu,cov, ef, rf)

start_time = time.time()
#N = ef.n_assets #len(tickerList)  # Number of assets
(muR,sR) = Resampling_EF(mu, cov,((-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1)),rf)
print(muR)
print(sR)
print("--- %s seconds ---" % (time.time() - start_time))


import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input
import plotly.graph_objs as go

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='ResampledEF',
                 figure={
                    'data':[
                            go.Scatter(
                                        x = sR,
                                        y = muR,
                                        mode = 'lines+markers',
                                        name = 'Resampled Efficient Frontier'),
                            go.Scatter(
                                        x = sigma_f2,
                                        y = mu_f2,
                                        mode = 'lines+markers',
                                        name = 'MV Efficient Frontier')

                                ],
                     'layout': go.Layout(yaxis=go.layout.YAxis(title='Means'),
                    xaxis=go.layout.XAxis(title='Standard Deviation'))
                 })
   # html.A('Download CSV', id='my-link'),

])

if __name__ == '__main__':

    app.run_server(debug=True)
'''
