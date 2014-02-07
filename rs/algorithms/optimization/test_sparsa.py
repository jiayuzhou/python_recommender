'''
Test the SpaRSA solver.

Created on Feb 5, 2014

@author: jiayu.zhou
'''
import timeit;
import numpy as np;
from rs.algorithms.optimization.sparsa import Opt_SpaRSA; 

def prox_l1(lamb, x, t = None):
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
    if t: # if step length exists, perform projection.
        tq = t * lamb;
        s  = 1 - np.minimum(tq/abs(x), 1);
        x = np.multiply(s, x);
    
    v = lamb * np.sum(np.abs(x));
    
    return [v, x];

def least_squares(w, X, y):
    '''
    least squares loss. 
    MATLAB verified function. 
    '''
    Xw_y = np.dot(X, w) - y;
    
    f = 0.5 * np.linalg.norm(Xw_y, 'fro')**2;
    g = np.dot(X.T, Xw_y); 
    g = g.reshape(g.shape[0] * g.shape[1] , 1, order = 'F');
    return [f, g];

if __name__ == '__main__':
    n = 500;
    d = 5000;
    X = np.mat(np.random.randn(n, d))
    y = np.mat(np.random.randn(n, 1));
    
#     n = 4;
#     d = 4;
#     X = np.mat([[0.1, 0.2, 0.3, 0.4], \
#                 [0.5, 0.6, 0.7, 0.8], \
#                 [0.9, 1.0, 1.1, 1.2], \
#                 [1.3, 1.4, 1.5, 1.6],
#                 [1.7, 1.8, 1.9, 2.0]])
#     y = np.mat([[-0.1], [-0.2], [0], [+ 0.1], [+0.2]]);
    
    lamb = 0.5;
    
    smoothF = lambda w: least_squares(w, X, y);
    nonsmoothF = lambda *args: prox_l1(0.5, *args);
    
    optimizer = Opt_SpaRSA(verbose = 10);
    
    #x0 = np.mat(np.zeros((d, 1)));
    #x0 = np.mat(np.random.randn(d, 1));
    x0 = np.mat(np.ones((d, 1)));
    
    tic = timeit.default_timer();
    optimizer.optimize(smoothF, nonsmoothF, x0);
    toc = timeit.default_timer();
    elapsed = toc - tic;
    print 'Elapsed time is ', str(elapsed);
