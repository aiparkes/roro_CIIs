import platform
import collections
from datetime import datetime
import functools
#from keras import backend as K
import numpy as np
import pandas as pd

def multiline(xs, ys, c, ax=None, **kwargs):
    """Plot lines with different colorings

    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection

    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

    Returns
    -------
    lc : LineCollection instance.
    """
    ax = plt.gca() if ax is None else ax
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)
    lc.set_array(np.asarray(c))
    ax.add_collection(lc)
    ax.autoscale()
    return lc


def fileDelim():
	#returns file delimiter appropriate for operating system
	#'Darwin' is most updated version of MacOS on 18/01/2019
	if platform.system() == 'Darwin':
		delim = '/'
	elif platform.system() == 'Linux':
		delim = '/'
	elif platform.system() == 'Windows':
		delim = '\\'
	return delim

def dataLocation():
	#will need changing for Silverstreams filesystem
	#output location of data file for my mac or linux computer, or windows laptop
	if platform.system() == 'Darwin':
		location = '/Users/amyparkes/18-month-data/'
	elif platform.system() == 'Linux':
		location = ''#'/home/amyparkes/Documents/Documents/Data/'#oe_paper/'#'/home/aip1u17/Data/'#
	elif platform.system() == 'Windows':
		location = 'C:\\Users\\aip1u17\\Documents\\Data\\oe_paper\\'
	return location
'''
def codeLocation():
	if platform.system() == 'Darwin':
		location = homeLocation()+'SilverStream/'
	elif platform.system() == 'Linux':
		location =  homeLocation()+'SilverStream/'
	elif platform.system() == 'Windows':
		location = homeLocation()+'SilverStream\\'
	return location
'''
def homeLocation():
	#will need changing for Silverstreams filesystem
	#output location of home file for my mac or linux computer, or windows laptop
	if platform.system() == 'Darwin':
		location = '/Users/amyparkes/'
	elif platform.system() == 'Linux':
		location ='/home/amyparkes/TEMP/TEMP/'#amyparkes/FileStore/mydocuments/'#'/home/aip1u17/'#'
	elif platform.system() == 'Windows':
		location = 'A:\\mydocuments\\'
	return location

def graphLocation():
	#will need changing for Silverstreams filesystem
	#output location of graph file for my mac or linux computer, or windows laptop
	if platform.system() == 'Darwin':
		location = '/Users/amyparkes/18_month_Graphs/'
	elif platform.system() == 'Linux':
		location ='/home/amyparkes/TEMP/TEMP/NetworkGraphs/'#amyparkes/FileStore/mydocuments/18_month_graph/'#'/home/aip1u17/'#
	elif platform.system() == 'Windows':
		location = homeLocation()+'18_month_graph\\'
	return location

def flatten(d, parent_key='', sep='_'):
	#convert config of keras object into dictionary items for printing
	items = []
	for k, v in d.items():
		new_key = parent_key + sep + k if parent_key else k
		if isinstance(v, collections.MutableMapping):
			items.extend(flatten(v, new_key, sep=sep).items())
		else:
			items.append((new_key, v))
	return dict(items)

def timer(func):
	#decorator to time func, returns function output and time elapsed
	@functools.wraps(func)
	def wrapper(*args, **kwargs):
		start = datetime.now()
		thing = func(*args, **kwargs)
		elapsed = datetime.now() - start
		return(thing, elapsed)
	return wrapper

def incaseCsvOpen(func):
	#decorator to handle permission error caused by attempting to write to an open csv in Windows
	#will try new filenames (same as function argument with a seeded random number before the '.csv')
	#until can write to one of them
	#NOTE: if trying to run this in a filesystem where you do not have write permission:
	#   THIS WILL CAUSE AN INFINITE LOOP
	#   please don't do this
	@functools.wraps(func)
	def wrapper(*args, **kwargs):
		np.random.seed(1234)
		printed = None
		fileName, fileType = kwargs['fileName'].split('.')
		while printed is None:
			try:
				func(*args, **kwargs)
			except PermissionError:
				print('Please don\'t leave the csv file you are trying to write to open.')
				rand_ish_num = np.random.randint(1,100)
				new_fileName = fileName+str(rand_ish_num)+'.'+fileType
				kwargs['fileName'] = new_fileName
				print('New csv file is called ',new_fileName)
			else:
				printed = True
		return
	return wrapper

def reset_weights(model):
	#reinitialised weights in model
	session = K.get_session()
	for layer in model.layers:
		for v in layer.__dict__:
			v_arg = getattr(layer,v)
			if hasattr(v_arg, 'initializer'):
				initializer_method = getattr(v_arg,'initializer')
				initializer_method.run(session=session)
	return

def l1_reg(weight_matrix):
	#regulariser, does NOT regularise ie NO restrictions imposed by this function
	return 0*K.sum(K.abs(weight_matrix))
	#return 0.000000000000001* K.sum(K.abs(weight_matrix))

def return_init(seed=None, mu=0, sigma=0.1):
	#returns my initialisation function, takes random seed, mean and std as inputs
	def my_init(shape, dtype=None):
		#my initialisation function
		#takes shape to fill with random numbers of mean mu and std sigma
		if seed:
			   np.random.seed(seed)
		init = np.random.normal(mu, sigma, shape)
		return init
	return my_init

def muStd(data):
	#returns mean and standard deviation of a dataset
	mu = np.mean(data)
	zeroMeanData = np.array(data) - mu
	var = np.sum(np.array(zeroMeanData)*np.array(zeroMeanData))/len(zeroMeanData)
	return(mu, np.sqrt(var))

def minMax(data):
	#returns min and max of a dataset
	minimum = min(data)
	maximum = max(data)
	return minimum, maximum

def normaliseMatrix(keys, keyStrings, scale=False):
	#normalises a dataset, either by scaling or with mean and std
	#handles pandas dataframes and lists/numpy arrays
	#returns normalised dataset
	#   mu and std of each variable and min and max of each variable, in dictionaries
	mu_std_dict = collections.OrderedDict()
	min_max_dict = collections.OrderedDict()
	if isinstance(keys, pd.DataFrame):
		new_keys = pd.DataFrame()
		for i, keyString in enumerate(keys.columns):
			mu, std = muStd(keys[keyString].astype(float).values)
			if std == 0:
				std = 0.00001
			mu_std_dict.update({keyString:(mu,std)})
			min, max = minMax(keys[keyString].astype(float).values)
			min_max_dict.update({keyString:(min,max)})
			if scale:
				new_keys[keyString] = scaleData(keys[keyString].astype(float).values, min, max)
			else:
				new_keys[keyString] = normData(keys[keyString].astype(float).values, mu, std)
	else:
		new_keys = np.ones(np.shape(keys))
		for i, keyString in enumerate(keyStrings):
			mu, std = muStd(keys[i,:])
			if std == 0:
				std = 0.00001
			min, max = minMax(keys[i,:])
			mu_std_dict.update({keyString:(mu,std)})
			min_max_dict.update({keyString:(min,max)})
			if scale:
				new_keys[i,:] = scaleData(keys[i,:], min, max)
			else:
				new_keys[i,:] = normData(keys[i,:], mu, std)
	return new_keys, mu_std_dict, min_max_dict

def unNormaliseMatrix(keys, keyStrings, min_max_dict, scale=False):
	#normalises a dataset, either by scaling or with mean and std
	#handles pandas dataframes and lists/numpy arrays
	#returns normalised dataset
	#   mu and std of each variable and min and max of each variable, in dictionaries
	if isinstance(keys, pd.DataFrame):
		new_keys = pd.DataFrame()
		for i, keyString in enumerate(keys.columns):
			min = min_max_dict[keyString][0]
			max = min_max_dict[keyString][1]
			if scale:
				new_keys[keyString] = unScaleData(keys[keyString].values, min, max)
			else:
				new_keys[keyString] = unNormData(keys[keyString].values, mu, std)
	else:
		new_keys = np.ones(np.shape(keys))
		for i, keyString in enumerate(keyStrings):
			min = min_max_dict[keyString][0]
			max = min_max_dict[keyString][1]
			if len(keyStrings) == 1:
				if scale:
					new_keys= unScaleData(keys, min, max)
				else:
					new_keys = unNormData(keys, mu, std)
			else:
				if scale:
					new_keys[:,i] = unScaleData(keys[:,i], min, max)
				else:
					new_keys[:,i] = unNormData(keys[:,i], mu, std)
	return new_keys

'''
def unNormaliseMatrix(keys, keyStrings, mu_std_dict):
	#takes normalised matrix and unnormalises
	new_keys = np.ones(np.shape(keys))
	for i, keyString in zip(range(len(keys)),keyStrings):
		new_keys[i,:] = unScaleData(keys[i,:], mu_std_dict[keyString][0], mu_std_dict[keyString][1])
	return new_keys
'''
def normData(vector, mu, std):
	#normalises a dataset
	zeroMeanData = vector - mu
	return zeroMeanData/std

def unNormData(vector, mu, std):
	#un normalises a dataset
	zeroMeanData = vector*std
	return zeroMeanData + mu

def scaleData(vector, min, max):
	#scales a dataset
	zeroMeanData = vector - min
	return zeroMeanData/(max-min)

def unScaleData(vector, min, max):
	#un scales a dataset
	zeroMeanData = vector*(max-min)
	return zeroMeanData + min

def polyFit0(x_vals, means):
	coeffs = np.polyfit(x_vals, means, 0)
	line = lambda x : coeffs[0]
	return line

def polyFit1(x_vals, means):
	((coeffs), (residuals), rank, (sing), cond)  = np.polyfit(x_vals, means, 1, full=True)
	line = lambda x : x*coeffs[0] + coeffs[1]
	return line, residuals[0]

def polyFit2(x_vals, means):
	((coeffs), (residuals), rank, (sing), cond)  = np.polyfit(x_vals, means, 2, full=True)
	line = lambda x : x*x*coeffs[0] + x*coeffs[1] + coeffs[2]
	return line, residuals[0]

def polyFit3(x_vals, means):
	((coeffs), (residuals), rank, (sing), cond)  = np.polyfit(x_vals, means, 3, full=True)
	line = lambda x : x*x*x*coeffs[0] + x*x*coeffs[1] + x*coeffs[2] + coeffs[3]
	return line, residuals[0]

def polyFit4(x_vals, means):
	((coeffs), (residuals), rank, (sing), cond)  = np.polyfit(x_vals, means, 4, full=True)
	line = lambda x : x*x*x*x*coeffs[0] + x*x*x*coeffs[1] + x*x*coeffs[2] + x*coeffs[3] + coeffs[4]
	return line, residuals[0]

def polyFit5(x_vals, means):
	((coeffs), (residuals), rank, (sing), cond)  = np.polyfit(x_vals, means, 5, full=True)
	line = lambda x : x*x*x*x*x*coeffs[0] + x*x*x*x*coeffs[1] + x*x*x*coeffs[2] + x*x*coeffs[3] + x*coeffs[4] + coeffs[5]
	return line, residuals[0]

def polyFit6(x_vals, means):
	((coeffs), (residuals), rank, (sing), cond)  = np.polyfit(x_vals, means, 6, full=True)
	line = lambda x : x*x*x*x*x*x*coeffs[0] + x*x*x*x*x*coeffs[1] + x*x*x*x*coeffs[2] + x*x*x*coeffs[3] + x*x*coeffs[4] + x*coeffs[5] + coeffs[6]
	return line, residuals[0]

def polyFit7(x_vals, means):
	((coeffs), (residuals), rank, (sing), cond)  = np.polyfit(x_vals, means, 7, full=True)
	line = lambda x : x*x*x*x*x*x*x*coeffs[0] + x*x*x*x*x*x*coeffs[1] + x*x*x*x*x*coeffs[2] + x*x*x*x*coeffs[3] + x*x*x*coeffs[4] + x*x*coeffs[5] + x*coeffs[6] + coeffs[7]
	return line, residuals[0]

def polyFit9(x_vals, means):
	coeffs = np.polyfit(x_vals, means, 10)
	line = lambda x : x*x*x*x*x*x*x*x*x*coeffs[0] + x*x*x*x*x*x*x*x*coeffs[1] + x*x*x*x*x*x*x*coeffs[2] + x*x*x*x*x*x*coeffs[3] + x*x*x*x*x*coeffs[4] + x*x*x*x*coeffs[5] + x*x*x*coeffs[6] + x*x*coeffs[7] + x*coeffs[8] + coeffs[9]
	return line

def bestPoly(x_vals, means):
	coeffs, residuals, rank, sing, cond = np.zeros((10,11)), np.zeros(10), np.zeros(10),np.zeros((10,11)),np.zeros(10)

	for i in range(10):
		((coeffs[i,0:i+1]), (residuals[i]), rank[i], (sing[i,0:i+1]), cond[i]) = np.polyfit(x_vals, means, i, full=True)
		if i > 1 and residuals[i] > residuals[i-1]:
			size = i-1
			print('degree ',size, 'coefficients ',coeffs[i-1,:])
			break
		elif i > 1 and residuals[i-1]/residuals[i] < 1.25:
			size = i
			print('degree ',size, 'coefficients ',coeffs[i,:])
			break
		elif i == 9:
			size = 9

	if size == 0:
		line = polyFit0(x_vals,means)
	if size == 1:
		line = polyFit1(x_vals,means)
	if size == 2:
		line = polyFit2(x_vals,means)
	if size == 3:
		line = polyFit3(x_vals,means)
	if size == 4:
		line = polyFit4(x_vals,means)
	if size == 5:
		line = polyFit5(x_vals,means)
	if size == 6:
		line = polyFit6(x_vals,means)
	if size == 7:
		line = polyFit7(x_vals,means)
	if size == 8:
		line = polyFit8(x_vals,means)
	if size == 9:
		line = polyFit9(x_vals,means)
	return line, i

def find_nearest(array, value):
	array = np.asarray(array)
	idx = (np.abs(array - value)).argmin()
	return idx


if __name__ == '__main__':
	main()
