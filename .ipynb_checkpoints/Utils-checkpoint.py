import numpy as np
from numpy.linalg import LinAlgError
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import probplot, moment
from statsmodels.tsa.stattools import acf, q_stat, adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.tsa.api as tsa
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARMA

def plot_correlogram(x, lags=None, title=None):
    lags = min(10, int(len(x)/5)) if lags is None else lags
    with sns.axes_style('whitegrid'):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 8))
        x.plot(ax=axes[0][0], title='Residuals')
        x.rolling(21).mean().plot(ax=axes[0][0], c='k', lw=1)
        q_p = np.max(q_stat(acf(x, nlags=lags), len(x))[1])
        stats = f'Q-Stat: {np.max(q_p):>8.2f}\nADF: {adfuller(x)[1]:>11.2f}'
        axes[0][0].text(x=.02, y=.85, s=stats, transform=axes[0][0].transAxes)
        probplot(x, plot=axes[0][1])
        mean, var, skew, kurtosis = moment(x, moment=[1, 2, 3, 4])
        s = f'Mean: {mean:>12.2f}\nSD: {np.sqrt(var):>16.2f}\nSkew: {skew:12.2f}\nKurtosis:{kurtosis:9.2f}'
        axes[0][1].text(x=.02, y=.75, s=s, transform=axes[0][1].transAxes)
        plot_acf(x=x, lags=lags, zero=False, ax=axes[1][0])
        plot_pacf(x, lags=lags, zero=False, ax=axes[1][1])
        axes[1][0].set_xlabel('Lag')
        axes[1][1].set_xlabel('Lag')
        fig.suptitle(title, fontsize=14)
        sns.despine()
        fig.tight_layout()
        fig.subplots_adjust(top=.9)
        

def log_transform(x):
    try:
        return np.log(x)
    except ValueError as e:
        print("0 element exist")
        
        
def optimal_ARMA_lags(train_size,up_bound_p,up_bound_q,y):
    # train_size (int) 
    # up_bound_p (int) : highest p to test for AR
    # up_bound_q (int) : highest q to test for MA
    # y (series) : time series
    # freq ('B','D','W','M','A','Q') : frequency of time series
    # choosing_criteria : options RMSE,BIC,AIC
    
    y_true = y[train_size:]
    results = {}
    
    for p in range(up_bound_p):
        for q in range(up_bound_q):
            aic,bic = [],[]
            if p == 0 and q == 0:
                continue
            print('ARMA({},{}) is being tested'.format(p,q))
            
            convergence_error = stationarity_error = 0
            y_pred = []
            
            for T in range(train_size,len(y)):
                train_set = y[T-train_size:T]
                try:
                    model = ARMA(endog=train_set,order=(p,q)).fit()
                except LinAlgError:
                    convergence_error += 1
                except ValueError:
                    stationarity_error += 1
                    
                forecast, _, _ = model.forecast(steps = 1)
                y_pred.append(forecast[0])
                aic.append(model.aic)
                bic.append(model.bic)
                
            result = (pd.DataFrame({'y_true':y_true,'y_pred':y_pred}).replace(np.inf,np.nan).dropna())
            rmse = np.sqrt(mean_squared_error(y_true = result.y_true,y_pred=result.y_pred))
            
            results[(p,q)] = {'Residual Mean Squared Error':rmse,'AIC':np.mean(aic),'BIC':np.mean(bic),'Convergence Error':convergence_error,'Stationarity Error':stationarity_error}
            
    arma_results = pd.DataFrame(results).T
    arma_results.columns = ['RMSE', 'AIC', 'BIC', 'convergence', 'stationarity']
    arma_results.index.names = ['p', 'q']
    
    return arma_results