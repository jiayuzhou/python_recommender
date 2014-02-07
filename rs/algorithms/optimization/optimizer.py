'''
Created on Feb 5, 2014

@author: jiayu.zhou
'''
import numpy as np;

class ProxOptimizer(object):
    
    FLAG_OPTIM   = 1;
    FLAG_XTOL    = 2;
    FLAG_FTOL    = 3;
    FLAG_MAXITER = 4;
    FLAG_MAXFEV  = 5;
    FLAG_OTHER   = 6;
  
    MESSAGE_OPTIM   = 'Optimality below optim_tol.';
    MESSAGE_XTOL    = 'Relative change in x below xtol.';
    MESSAGE_FTOL    = 'Relative change in function value below ftol.';
    MESSAGE_MAXITER = 'Max iterations reached.';
    MESSAGE_MAXFEV  = 'Max function evaluations reached.';
    

def proximal(f, prox_f):
    '''
    construct a proximal operand used for proximal methods, e.g. sparsa. 
    prox_f( y, t ) = argmin_X 1/(2*t)*|| x - y ||^2 + f(x)
                   = argmin_X 1/2 *|| x - y ||^2 + t * f(x)
                   
    e.g. to solve l1 norm. 
         f(x)         = lam * np.norm(x,'fro');
         prox_f(x, t) = np.multiply(x, (1 - np.minimum( (t * lamb)/abs(x), 1 )));
    '''
    
    def fcn_impl(x, t = None):
        if t: # if step length exists, perform projection.
            x = prox_f(x, t);
        v = f(x);
        return [v, x];
    
    return fcn_impl;



def prox_l1(lamb):
    '''
    The proximal function of l1 norm regularization.
    The function is constructed from proximal(f, prox_f).
    
    Parameters
    ----------
    lamb: the l1 regularization.   
    '''
    # function value.
    gx    = lambda x : lamb * np.linalg.norm(x, 'fro');
    # proximal gradient. 
    # prox_f( y, t ) = argmin_X 1/(2*t)*|| x - y ||^2 + lamb * |x|
    gprox = lambda x, t :  np.multiply(x, (1 - np.minimum( (t * lamb)/abs(x), 1 )));
    # construct proximal methods.  
    return proximal(gx, gprox);

