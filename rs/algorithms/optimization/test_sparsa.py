'''
Test the SpaRSA solver.

Created on Feb 5, 2014

@author: jiayu.zhou
'''

import numpy as np;

def prox_l1(lamb, x, t):
    '''
    l1 projection. 
    
    Parameters
    ----------
    lamb: regularization parameter
    x: point to be projected. 
    t: step size
    
    Returns
    ----------
    v: value
    x: projection. 
    t: step size. 
    
    '''
    v = lamb * sum(abs(x));
    
    tq = t * lamb;
    s  = 1 - min(tq/abs(x), 1);
    x = np.multiply(s, x);
    
    return [v, x];

def least_squares(w, X, y, compute_gradient = False):
    '''
    least squares loss. 
    '''
    Xw_y = np.dot(X, w) - y;
    
    f = 0.5 * np.linalg.norm(Xw_y, 'fro');
    if not compute_gradient:
        return [f];
    g = np.dot(X.T, Xw_y); 
    return [f, g];

if __name__ == '__main__':
    
    


