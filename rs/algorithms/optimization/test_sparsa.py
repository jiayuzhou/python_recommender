'''
Test the SpaRSA solver.

Created on Feb 5, 2014

@author: jiayu.zhou
'''
import timeit;
import numpy as np;
from rs.algorithms.optimization.sparsa import Opt_SpaRSA; 
from rs.algorithms.optimization.prox import prox_l1
from rs.algorithms.optimization.smooth import least_squares;
from rs.algorithms.optimization.prox import proj_nonneg_simplex;

if __name__ == '__main__':
#     n = 500;
#     d = 5000;
#     X = np.mat(np.random.randn(n, d))
#     y = np.mat(np.random.randn(n, 1));
    
    n = 4;
    d = 4;
    X = np.mat([[0.1, 0.2, 0.3, 0.4], \
                [0.5, 0.6, 0.7, 0.8], \
                [0.9, 1.0, 1.1, 1.2], \
                [1.3, 1.4, 1.5, 1.6],
                [1.7, 1.8, 1.9, 2.0]])
    y = np.mat([[-0.1], [-0.2], [0], [+ 0.1], [+0.2]]);
    
    lamb = 0.1;
    
    smoothF = lambda w: least_squares(w, X, y);
    nonsmoothF = prox_l1(lamb);
    
    optimizer = Opt_SpaRSA(verbose = 10);
    
    #x0 = np.mat(np.zeros((d, 1)));
    #x0 = np.mat(np.random.randn(d, 1));
    x0 = np.mat(np.ones((d, 1)));
    
    tic = timeit.default_timer();
    [xopt, _, _] = optimizer.optimize(smoothF, nonsmoothF, x0);
    toc = timeit.default_timer();
    elapsed = toc - tic;
    print 'Elapsed time is ', str(elapsed);
    
    print xopt;
    
    # sparsity.
    nnz = float(np.sum(xopt!=0));
    print 'Density: ', str(nnz/d);
    
    
    print '>>test proj_nonneg_simplex with sparsa '
    
    n = 500;
    d = 5000;
    X = np.mat(np.random.randn(n, d))
    y = np.mat(np.random.randn(n, 1));
    
    smoothF = lambda w: least_squares(w, X, y);
    nonsmoothF = proj_nonneg_simplex(1);
    
    optimizer = Opt_SpaRSA(verbose = 10);
    x0 = np.mat(np.ones((d, 1)));
    [xopt, _, _] = optimizer.optimize(smoothF, nonsmoothF, x0);
    
    print 'Sum:' 
    print np.sum(xopt);
    
    nnz = float(np.sum(xopt!=0));
    print 'Density: ', str(nnz/d);