'''

Some utilities of sparse matrix. 

Created on Jan 29, 2014

@author: jiayu.zhou
'''

import scipy.sparse as sp

import math


def normalize_row(sparse_mat, minf = 0.0001):
    '''
    Normalize a sparse matrix in a row fashion (the sum of each row is 1).
    
    
    This procedure follows the answer in this page:
    http://stackoverflow.com/questions/12305021/efficient-way-to-normalize-a-scipy-sparse-matrix
    '''
    
    # perform operations in lil_matrix. 
    sparse_mat = sparse_mat.tolil(); 

    # work on the transpose (and we normalize each column on the transposed matrix).
    sparse_mat = sparse_mat.T;
    
    sum_of_col = sparse_mat.sum(0).tolist();
    c = [];
    for i in sum_of_col:
        for j in i:
            if math.fabs(j)<minf:
                c.append(0)
            else:
                c.append(1/float(j))
                
    B = sp.lil_matrix((len(c), len(c)));
    B.setdiag(c)
    
    sparse_mat = sparse_mat * B;
    
    return sparse_mat.T;


