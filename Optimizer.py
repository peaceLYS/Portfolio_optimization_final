'''This is all optimizer we realized
  Created by Chaoyi Ye and some functions credit to Chaoyi Ye
  Some functions credit to Hangchi Li and added by Yishi Liu
  
'''

import numpy as np
import math
from scipy.stats.mstats import gmean

import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
from scipy.optimize import minimize
#yf.pdr_override()

#***** pip install PyPortfolioOpt ******
from pypfopt import risk_models
#from pypfopt.hierarchical_risk_parity import hrp_portfolio
from empyrical import downside_risk,sortino_ratio,calmar_ratio,omega_ratio,tail_ratio,max_drawdown,annual_volatility,sharpe_ratio,value_at_risk,conditional_value_at_risk,down_capture,up_capture
from empyrical.stats import alpha_beta_aligned,stability_of_timeseries
'''This function uses PyPortfolioOpt to do Mean-Variance Optimization
@param Goal: an input_value, Optimization_Goal
@return optimized weights, Expected Return, Annual Volatility, Sharpe Ratio
EfficientFrontier object is created outside of this function
'''
def Optimization(Goal, ef=None , vr = None,rp=None, rf = 0.02, target_return=0.20, target_risk=0.20):
    weights = []
    if Goal == '1.Maximize Sharpe Ratio' :
        weights = ef.max_sharpe(rf)
    elif Goal == '6.Minimize Volatility subject to...' : #Minimize Volatility subject to...
        weights = ef.efficient_return(target_return, market_neutral=False)
    elif Goal == '2.Minimize Variance' :
        weights = ef.min_volatility()
    elif Goal == '7.Maximize Return subject to...' : #Maximize Return subject to...
        weights = ef.efficient_risk(rf,target_risk)
    #if ef is not None:
    #    (Er, Vol, Sharpe) = ef.portfolio_performance(verbose=True,risk_free_rate=rf)
    #if vr is not None:
    elif Goal == '3.Minimize Conditional Value-at-Risk':#Minimize Conditional Value-at-Risk
        print("I am at goal 3 MVOpt")
        weights = vr.min_cvar()
    elif Goal == '4.Risk Parity' :
        weights= rp.hrp_portfolio()
    
    return weights# Er, Vol, Sharpe

'''This All_frontier function calculates 20 points between min vol point
and max return point on the Efficient Frontier
@param m & cov : used to calculate expected return and volatility of the portfolio from Weights
@param ef : an EfficientFrontier object, comes with instance variables: n_assets, mu, cov
            and member functions to calculate optimal weights in different methods
@return portfolio weights, mu, sigma, arrays ready for graphing      '''
def All_frontier(m,cov,ef,rf=0.02):
    cov = np.matrix(cov)
    n = ef.n_assets  #num of assets
    avg_ret = np.matrix(m).T
    w_minV = ef.min_volatility()
    w_maxR = ef.efficient_risk(risk_free_rate=rf,target_risk=0.50)
    r_min = list(w_minV.values()) @ m
    r_max = list(w_maxR.values()) @ m
    points = 20
    interval = abs((r_max - r_min)/ points)
    mus = []
    for i in range(points):
        r_min += interval
        mus.append(r_min)

    pf_weights =[]
    pf_mu = []
    pf_sigma = []
    for yy in mus:
        weights = list(ef.efficient_return(target_return = abs(float(yy)) ).values())
        pf_weights.append(weights)
        #(mu, sigma, sharpe)=ef.portfolio_performance(verbose=False,risk_free_rate=rf)
        mu = weights @ m
        sigma = math.sqrt(weights @ cov @ weights)
        pf_mu.append(mu)
        pf_sigma.append(sigma)
    return pf_weights, pf_mu, pf_sigma


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

def Min_Max_drawdown(price,target_return,num_ticker,period):
    def total_weight_constraint(x):
        return np.sum(x)-1.0

    def long_only_constraint(x):
        return x
    def no_leverage(x):
        return 1-x
    x_cons = ({'type': 'eq', 'fun': total_weight_constraint},{'type': 'ineq', 'fun': long_only_constraint},
              {'type': 'ineq', 'fun': no_leverage})
       
    def getdata(period='M'):
        returns=calreturn(price,period)
        return returns
    
    def retrange(price,period='M'):
        returns=calreturn(price,period)
        returns_y=returns.mean()*12
        return returns_y.min(),returns_y.max()

    return_min,return_max=retrange(price)
    if (target_return>return_max) or (target_return<return_min):
        print(' Please reinput target_return')
    
    
    
    def getTarReturn(target_return):
        return target_return
    
    
    
    def weigdrawback(x0,target_return):
        a=getdata()
        b=getTarReturn(target_return)
        wd=list((a*x0).apply(lambda x:x.sum(),axis=1))
        diff=np.abs(np.mean(wd)*12-b)
        return diff
    x0=[x for x in range(0,num_ticker)]
    x0=x0/np.sum(x0)
    #cons=({'type': 'eq', 'fun': w1+w2+w3-1})
    y_mmd=minimize(weigdrawback,x0,args=target_return,constraints=x_cons)
    return y_mmd

def Max_Sortino_Ratio(price,target_return,num_ticker,bound,period):
    def total_weight_constraint(x):
        return np.sum(x)-1.0

    def long_only_constraint(x):
        return x
    def no_leverage(x):
        return 1-x
    x_cons = ({'type': 'eq', 'fun': total_weight_constraint},{'type': 'ineq', 'fun': long_only_constraint},
              {'type': 'ineq', 'fun': no_leverage})
       
    def getdata(period='M'):
        returns=calreturn(price,period)
        return returns
    
    
    
    def getTarReturn(target_return):
        return target_return
    
    
  
    
    def sortino_r(x0,target_return):
        a=getdata()
        b=getTarReturn(target_return)
        wd=list((a*x0).apply(lambda x:x.sum(),axis=1))
        sr=sortino_ratio(np.array(wd),required_return=b)
        return -sr
    x0=[x for x in range(0,num_ticker)]
    x0=x0/np.sum(x0)
    #cons=({'type': 'eq', 'fun': w1+w2+w3-1})
    y_sr=minimize(sortino_r,x0,args=target_return,constraints=x_cons)
    return y_sr

def Max_Omega_Ratio(price,target_return,rf,num_ticker,bound,period):
    def total_weight_constraint(x):
        return np.sum(x)-1.0

    def long_only_constraint(x):
        return x
    def no_leverage(x):
        return 1-x
    x_cons = ({'type': 'eq', 'fun': total_weight_constraint},{'type': 'ineq', 'fun': long_only_constraint},
              {'type': 'ineq', 'fun': no_leverage})
       
    def getdata(period='M'):
        returns=calreturn(price,period)
        return returns
    
    
    
    def getTarReturn(target_return):
        return target_return
    
    
  
    
    def omega_r(x0,target_return):
        a=getdata()
        b=getTarReturn(target_return)
        wd=list((a*x0).apply(lambda x:x.sum(),axis=1))
        omegar=omega_ratio(np.array(wd),risk_free=rf/12,required_return=b)
        return -omegar
    x0=[x for x in range(0,num_ticker)]
    x0=x0/np.sum(x0)
    #cons=({'type': 'eq', 'fun': w1+w2+w3-1})
    y_or=minimize(omega_r,x0,args=target_return,bounds=bound,constraints=x_cons)
    return y_or

def coskew(price,num_ticker,bound,period):
    def total_weight_constraint(x):
        return np.sum(x)-1.0

    def long_only_constraint(x):
        return x
    def no_leverage(x):
        return 1-x
    x_cons = ({'type': 'eq', 'fun': total_weight_constraint},{'type': 'ineq', 'fun': long_only_constraint},
              {'type': 'ineq', 'fun': no_leverage})
       
    def getdata(period='M'):
        returns=calreturn(price,period)
        return returns
    
    
    
    
  
    
    def co_skew(x0):
        a=getdata()
        #b=getTarReturn(target_return)
        wd=list((a*x0).apply(lambda x:x.sum(),axis=1))
        df=pd.DataFrame()
        df['portfolio']=np.array(wd)
        return -df.skew()
    x0=[x for x in range(0,num_ticker)]
    x0=x0/np.sum(x0)
    #cons=({'type': 'eq', 'fun': w1+w2+w3-1})
    y_coskew=minimize(co_skew,x0,bounds=bound,constraints=x_cons)
    return y_coskew

def cokurt(price,num_ticker,bound,period):
    def total_weight_constraint(x):
        return np.sum(x)-1.0

    def long_only_constraint(x):
        return x
    def no_leverage(x):
        return 1-x
    x_cons = ({'type': 'eq', 'fun': total_weight_constraint},{'type': 'ineq', 'fun': long_only_constraint},
              {'type': 'ineq', 'fun': no_leverage})
       
    def getdata(period='M'):
        returns=calreturn(price,period)
        return returns
    
    
    
    
  
    
    def co_kurt(x0):
        a=getdata()
        #b=getTarReturn(target_return)
        wd=list((a*x0).apply(lambda x:x.sum(),axis=1))
        df=pd.DataFrame()
        df['portfolio']=np.array(wd)
        return np.abs(df.kurt())
    x0=[x for x in range(0,num_ticker)]
    x0=x0/np.sum(x0)
    #x0=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0,0,0]
    #cons=({'type': 'eq', 'fun': w1+w2+w3-1})
    y_cokurt=minimize(co_kurt,x0,bounds=bound,constraints=x_cons)
    return y_cokurt

def tailriskparity(price,num_ticker,bound):
    def total_weight_constraint(x):
        return np.sum(x)-1.0

    def long_only_constraint(x):
        return x
    def no_leverage(x):
        return 1-x
    x_cons = ({'type': 'eq', 'fun': total_weight_constraint},{'type': 'ineq', 'fun': long_only_constraint},
              {'type': 'ineq', 'fun': no_leverage})
     
          
    
    def calculate_cvar(w,price):
        rp=w*calreturn(price,'M')
        l=len(rp)
        # ascending=True
        CVaR=-rp.apply(lambda x:x.sum(),axis=1).sort_values()[:round(l*0.95)].sum()/round(l*0.95)
        return np.matrix(CVaR)[0,0]  
    
    def calculate_risk_contribution(w,price):
        returns=calreturn(price,'M')
        l=len(returns)
        MRC=[]
        CVaR=calculate_cvar(w,price)
        for col in returns.columns:
            MRC.append(-returns[col].sort_values()[:round(l*0.95)].sum()/round(l*0.95))
        RC=np.multiply(np.matrix(MRC),np.matrix(w))/CVaR
        return RC 
    
    
    def risk_budget_objective(x,pars):
        # calculate portfolio risk
        price = pars[0]# covariance table
        x_t = pars[1] # risk target in percent of portfolio risk
        CVaR=calculate_cvar(x,price)
        #sig_p =  np.sqrt(calculate_portfolio_var(x,V)) # portfolio sigma
        risk_target = np.asmatrix(np.multiply(CVaR,x_t))
        asset_RC = calculate_risk_contribution(x,price)
        J = np.sum(np.square(asset_RC-risk_target)) # sum of squared error
        return J  
        
    
    x0=[x for x in range(0,num_ticker)]
    x0=x0/np.sum(x0)
    x_t=[1 for x in range(0,num_ticker)]
    x_t=x_t/np.sum(x_t)
    #cons=({'type': 'eq', 'fun': w1+w2+w3-1})
    res= minimize(risk_budget_objective, x0, args=[price,x_t], bounds=bound,method='SLSQP',constraints=x_cons,options={'disp': True})
    return res 

def co_drawdown(price,num_ticker,bound):
    def total_weight_constraint(x):
        return np.sum(x)-1.0

    def long_only_constraint(x):
        return x
    def no_leverage(x):
        return 1-x
    x_cons = ({'type': 'eq', 'fun': total_weight_constraint},{'type': 'ineq', 'fun': long_only_constraint},
              {'type': 'ineq', 'fun': no_leverage})
     
        
    def calculate_mdd(w,price):
        price_m=price.apply(lambda x:x.resample('M').last())
        rp=(w*price_m).apply(lambda x:x.sum(),axis=1)
        l=len(rp)
        mdd=[]
        start=[]
        end=[]
    
        for i in range(0,l):
            if rp[i]>np.min(rp[i+1:]):
                mdd.append((np.min(rp[i+1:])-rp[i])/rp[i])
                start.append(rp.index[i])
                end.append(rp.loc[rp==np.min(rp[i+1:])].index[0])
    
            
        # ascending=True
        max_drawdown=np.min(mdd)
        num=mdd.index(max_drawdown)
        return max_drawdown,start[num],end[num] 
    
    def cal_RC(w,price):
        max_dd,start,end=calculate_mdd(w,price)
        price_m=price.apply(lambda x:x.resample('M').last())
        rp=(w*price_m).apply(lambda x:x.sum(),axis=1)
        MRC=[]
        for col in price_m.columns:
            MRC.append((rp[end]*price_m[col][start]-rp[start]*price_m[col][end])/(rp[start]**2))
    
        return MRC
    
    
    def risk_budget_objective(w,price):
        return np.abs(np.mean(cal_RC(w,price)))
    
    x0=[x for x in range(0,num_ticker)]
    x0=x0/np.sum(x0)
    
    #cons=({'type': 'eq', 'fun': w1+w2+w3-1})
    res= minimize(risk_budget_objective, x0, args=price,bounds=bound, method='SLSQP',constraints=x_cons,options={'disp': True})
    return res  

def Infor_Ratio(price,Benchmark,num_ticker,bound,period):
    def total_weight_constraint(x):
        return np.sum(x)-1.0

    def long_only_constraint(x):
        return x
    def no_leverage(x):
        return 1-x
    x_cons = ({'type': 'eq', 'fun': total_weight_constraint},{'type': 'ineq', 'fun': long_only_constraint},
              {'type': 'ineq', 'fun': no_leverage})
    
    def getdata(period='M'):
        returns=calreturn(price,period)
        return returns
    
    def getbenchmark(Benchmark,period='M'):
        first_m=Benchmark.resample(period).first()
        last_m=Benchmark.resample(period).last()
        benchm=last_m/first_m-1
        return benchm
    
    def infor(x0,benchmark):
        a=getdata()
        b=getbenchmark(benchmark)
        wd=list((a*x0).apply(lambda x:x.sum(),axis=1))
        ir=(np.mean(wd-b))/(np.std(wd-b))
        return -ir
    x0=[x for x in range(0,num_ticker)]
    x0=x0/np.sum(x0)
    #a=np.array(price)
    #cons=({'type': 'eq', 'fun': w1+w2+w3-1})
    y_ir=minimize(infor,x0,args=Benchmark,bounds=bound,constraints=x_cons)#bounds=[(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1)],
    return y_ir

def metrics(w,price,Benchmark):
    name=[]
    value=[]
    rp=(w*price).apply(lambda x:x.sum(),axis=1)
    rp_mr=rp.resample('M').last()/rp.resample('M').first()-1

    rp_yr=rp.resample('Y').last()/rp.resample('Y').first()-1

    # Arithmetic Mean
    name.append('Arithmetic Mean (monthly)')
    value.append(rp_mr.mean())

    name.append('Arithmetic Mean (annualized)')
    value.append(rp_yr.mean())

    #Geometric Mean
    name.append('Geometric Mean (monthly)')
    value.append(gmean(np.abs(rp_mr.dropna())))

    name.append('Geometric Mean (annualized)')
    value.append(gmean(np.abs(rp_yr.dropna())))

    #Volatility
    name.append('Volatility (monthly)')
    value.append(rp_mr.std())

    name.append('Volatility (annualized)')
    an_vola=annual_volatility(rp.pct_change().dropna(),period='daily')
    value.append(an_vola)

    #Downside_deviation
    name.append('Downside_deviation')
    value.append(downside_risk(returns=rp_mr))

    #Max. Drawdown
    name.append('Max_Drawdown')
    value.append(max_drawdown(rp_mr))

    #US Market Correlation
    name.append('US Market Correlation')
    corr=np.corrcoef(rp.pct_change().dropna(),Benchmark.pct_change().dropna())[0][1]
    value.append(corr)

    #Beta,alpha
    alpha,beta=alpha_beta_aligned(rp.pct_change().dropna(), Benchmark.pct_change().dropna(), risk_free=0.02, period='daily', annualization=None)
    name.append('beta')
    value.append(beta)

    name.append('alpha')
    value.append(alpha*12/100)

    #R square
    name.append('R²')
    R_Square=stability_of_timeseries(rp.pct_change().dropna())
    value.append(R_Square)

    #sharpe_ratio
    name.append('Sharpe_ratio')
    sharp_ratio=sharpe_ratio(rp.pct_change().dropna())
    value.append(sharp_ratio)

    #Sortino_Ratio
    name.append('Sortino_ratio')
    value.append(sortino_ratio(rp.pct_change().dropna()))

    #Treynor Ratio
    name.append('Treynor_ratio')
    T_R=sharp_ratio*an_vola/beta
    value.append(T_R)

    #calmar_ratio
    name.append('calmar_ratio')
    value.append(calmar_ratio(rp.pct_change().dropna()))

    # active return
    name.append('active return')
    active_return=(rp.pct_change().dropna()-Benchmark.pct_change().dropna()).mean()*252
    value.append(active_return)

    #tracking error
    name.append('tracking error')
    track_error=(rp.pct_change().dropna()-Benchmark.pct_change().dropna()).std()*np.sqrt(252)
    value.append(track_error)

    #information ratio
    name.append('Information ratio')
    ir=(np.mean(rp.pct_change().dropna()-Benchmark.pct_change().dropna()))*np.sqrt(252)/(np.std(rp.pct_change().dropna()-Benchmark.pct_change().dropna()))
    value.append(ir)

    #skewness
    name.append('Skewness')
    value.append(rp.pct_change().dropna().skew())

    #excess kurtosis
    name.append('Excess kurtosis')
    value.append(rp.pct_change().dropna().kurt()-3)

    #historical value at risk
    name.append('Historical Value-at-Risk (5%)')
    value.append(rp_mr.sort_values().quantile(0.05))

    #Conditional Value-at-Risk (5%)
    name.append('Conditional Value-at-Risk (5%)')
    value.append(conditional_value_at_risk(rp_mr))

    #Upside Capture Ratio (%)
    name.append('Upside Capture Ratio (%)')
    b_mr=Benchmark.resample('M').last()/Benchmark.resample('M').first()-1
    value.append(up_capture(rp_mr, b_mr)*100)

    #Downside Capture Ratio (%)
    name.append('Downside Capture Ratio (%)')
    value.append(down_capture(rp_mr, b_mr)*100)

    #Positive Periods
    name.append('Positive Periods')
    copy=np.maximum(rp_mr,0)
    win=1-len(copy.loc[copy==0])/len(copy)
    value.append(win)

    #Gain/Loss Ratio
    name.append('Gain/Loss Ratio')
    value.append(win/(1-win))

    return name,value
