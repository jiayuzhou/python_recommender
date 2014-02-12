'''
Created on Feb 12, 2014

@author: jiayu.zhou
'''

from scipy.sparse import csr_matrix;
import rs.data.data_split as ds; 
import numpy as np;
from rs.data.recdata import FeedbackData;

if __name__ == '__main__':
    data    = np.array([-0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 ]);
    indices = np.array([0,    2,   4,   0,   2,   3,   1,   2,   4]);
    indptr  = np.array([0,              3,             6,            9]);
    
    lo_data   = csr_matrix( (data,indices,indptr), shape=(3, 5)).tocoo();
    
    fb_data = FeedbackData(lo_data.row.tolist(), lo_data.col.tolist(), lo_data.data.tolist(), 3, 5,
                           np.array([]), np.array([]), np.array([]));
    
    print 'Original data:'
    print fb_data.get_sparse_matrix().todense();

#     leave_k_out_idx = {};
#     leave_k_out_idx [0] = set([4]);
#     leave_k_out_idx [1] = set([0, 3]);
#     leave_k_out_idx [2] = set([2]);
    
    # generate leave k indices. 
    leave_k_out_idx = ds.leave_k_out(fb_data, 2);
    
    print 'leave_k_indices'
    print leave_k_out_idx;
    
    [lo_data, tr_data] = fb_data.leave_k_out(leave_k_out_idx);
    
    print 'Leave k out:'
    print lo_data.get_sparse_matrix().todense();
    
    print 'Remaining:'
    print tr_data.get_sparse_matrix().todense();
    
    
    
    
    
    
    