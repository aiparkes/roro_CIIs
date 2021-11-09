import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

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
