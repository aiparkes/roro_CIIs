import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import csv_utils
import glob
import seaborn as sns
from scipy import stats
from collections import OrderedDict

def ols_reg(data):
	ols = smf.ols('LOG_AER ~ LOG_DWT', data).fit()
	ols_ci = ols.conf_int().loc['LOG_DWT'].tolist()
	ols = dict(a = ols.params['Intercept'],
			   b = ols.params['LOG_DWT'],
			   lb = ols_ci[0],
			   ub = ols_ci[1])
	return ols

def fit_model(q, mod):
	res = mod.fit(q=q)
	return [q, res.params['Intercept'], res.params['LOG_DWT']] + \
			res.conf_int().loc['LOG_DWT'].tolist()

def quant_reg(data, quantiles):
	mod = smf.quantreg('LOG_AER ~ LOG_DWT', data)
	models = [fit_model(x, mod) for x in quantiles]
	models = pd.DataFrame(models, columns=['q', 'a', 'b', 'lb', 'ub'])
	return models

def describe_distribution(data, name, set_stats):
	[nobs, (min, max), mean, variance, skewness, kurtosis] = stats.describe(data.AER)
	for stat_name, stat in zip(['nobs', 'min', 'max', 'mean', 'median', 'variance', 'skewness', 'kurtosis'],
					[nobs, min, max, mean, np.median(data.AER), variance, skewness, kurtosis]):
		set_stats.update({name+'_'+stat_name:stat})

	#set_stats['df'][name] =  [nobs, min, max, mean, np.median(data.AER), variance, skewness, kurtosis]
	return set_stats
