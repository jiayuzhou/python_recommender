'''
Created on Jan 29, 2014

@author: jiayu.zhou
'''
import scipy.sparse as sp
import numpy as np
import math
from rs.utils.sparse_matrix import normalize_row;
    
if __name__ == '__main__':    
    minf = 0.0001
    
    A = sp.coo_matrix((5,5))
    
    
    
    A = A.tolil();
    
    #A = sp.lil_matrix((5,5))
    b = np.arange(0,5)
    A.setdiag(b[:-1], k=1)
    A.setdiag(b)
    print 'Dense A:'
    print A.todense()
    
    C1 = normalize_row(A);
    
    
    
    A = A.T
    print 'Dense A transpose:'
    print A.todense()
    
    
    
    sum_of_col = A.sum(0).tolist()
    print sum_of_col
    c = []
    for i in sum_of_col:
        for j in i:
            if math.fabs(j)<minf:
                c.append(0)
            else:
                c.append(1/j)
    
    print c
    
    B = sp.lil_matrix((5,5))
    B.setdiag(c)
    print B.todense()
    
    C = A*B
    print C.todense()
    C = C.T
    print 'Procedure'
    print C.todense()
    
    print 'Function'
    print C1.todense()