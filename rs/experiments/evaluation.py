'''
Provides a list of evaluation metrics.

Created on Feb 3, 2014

@author: jiayu.zhou
'''

from itertools import izip;

def rmse(list1, list2):
    '''
    Return the root mean squared error.  
    sqrt(sum_i ((x[i] - y[i])**2))
    '''
    if not len(list1) == len(list2):
        raise ValueError('Dimension not match.');
    
    return (sum( (x - y) ** 2 for x, y in izip(list1, list2))) ** 0.5;