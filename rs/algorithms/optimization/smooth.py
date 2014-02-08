'''
Includes a set of (smooth) loss functions. 

Created on Feb 8, 2014

@author: jiayu.zhou
'''

import numpy as np;

def least_squares(w, X, y):
    '''
    least squares loss. 
    MATLAB verified function.
    
    f(x) = 1/2 * ||X * w - y||_F^2.
    
    Parameters
    ----------
    w: np.matrix 
    X: np.matrix 
    y: np.matrix 
    
    Returns
    ----------
    
    '''
    Xw_y = np.dot(X, w) - y;
    
    f = 0.5 * np.linalg.norm(Xw_y, 'fro')**2;
    g = np.dot(X.T, Xw_y); 
    g = g.reshape(g.shape[0] * g.shape[1] , 1, order = 'F');
    return [f, g];